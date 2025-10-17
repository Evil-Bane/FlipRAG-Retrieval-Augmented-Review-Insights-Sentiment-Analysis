# main.py (revised to include product description in RAG context)
"""
main.py

One-file runner:
- Scrapes up to 50 Flipkart reviews (max 5 pages)
- Performs sentiment analysis (per-review + aggregate)
- Builds FAISS index from review chunks (embeddings via sentence-transformers)
- Provides a browser GUI using Gradio:
  - Input Flipkart URL -> Scrape + Sentiment summary + Product description context
  - Chat box (RAG powered by Ollama + retrieved review chunks + product description)
"""
import os
import re
import json
import time
from typing import List, Dict, Optional, Tuple
import requests
from bs4 import BeautifulSoup

# NLP / embeddings / vectorstore / llm
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Gradio UI
import gradio as gr

# ---------------- Config ----------------
MAX_REVIEWS = 50
MAX_PAGES = 5
CHUNK_SIZE = 700
CHUNK_OVERLAP = 150
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # for embeddings
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"  # transformers pipeline
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = "dolphin-mistral:latest"

TOP_K = 6

# ---------------- Ollama wrapper ----------------
class OllamaClient:
    def __init__(self, api_url: str = OLLAMA_API_URL, model: str = OLLAMA_MODEL, timeout: int = 60):
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        url = f"{self.api_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {"temperature": temperature, "max_tokens": max_tokens},
            "stream": False
        }
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        # best-effort parsing (different Ollama versions differ)
        if isinstance(data, dict):
            if "choices" in data and isinstance(data["choices"], list):
                return "".join([c.get("text", "") for c in data["choices"]])
            if "output" in data:
                if isinstance(data["output"], list):
                    return "".join(data["output"])
                if isinstance(data["output"], str):
                    return data["output"]
            if "result" in data and isinstance(data["result"], str):
                return data["result"]
            if "text" in data and isinstance(data["text"], str):
                return data["text"]
            if "response" in data and isinstance(data["response"], str):
                return data["response"]
        return json.dumps(data)


# ---------------- Flipkart scraping (adapted) ----------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

def to_review_url(url: str) -> str:
    if "flipkart.com" not in url:
        raise ValueError("Not a Flipkart URL")
    url = url.split("?")[0]
    if "/product-reviews/" in url:
        return url
    m = re.search(r"flipkart\.com/(?:[^\s/]+(?:/[^/]+)*)/p/([A-Za-z0-9\-]+)", url)
    if m:
        pid = m.group(1)
        slug = url.split("/p/")[0].split("flipkart.com/")[-1].rstrip("/")
        return f"https://www.flipkart.com/{slug}/product-reviews/{pid}"
    parts = url.rstrip("/").split("/")
    if len(parts) >= 2:
        slug = parts[-2]
        pid = parts[-1]
        return f"https://www.flipkart.com/{slug}/product-reviews/{pid}"
    raise ValueError("Could not derive review URL from the provided Flipkart link")

### NEW: derive product URL (p/<id>) from input url or review url
def to_product_url(url: str) -> str:
    url = url.split("?")[0]
    if "/p/" in url:
        return url
    # if it's product-reviews, simply replace that segment
    if "/product-reviews/" in url:
        return url.replace("/product-reviews/", "/p/")
    # fallback similar to above logic
    m = re.search(r"flipkart\.com/(?:[^\s/]+(?:/[^/]+)*)/p/([A-Za-z0-9\-]+)", url)
    if m:
        pid = m.group(1)
        slug = url.split("/p/")[0].split("flipkart.com/")[-1].rstrip("/")
        return f"https://www.flipkart.com/{slug}/p/{pid}"
    parts = url.rstrip("/").split("/")
    if len(parts) >= 2:
        slug = parts[-2]
        pid = parts[-1]
        return f"https://www.flipkart.com/{slug}/p/{pid}"
    raise ValueError("Could not derive product URL from the provided Flipkart link")


def extract_total_pages(html: str) -> int:
    m = re.search(r"Page\s*\d+\s*of\s*(\d+)", html, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except:
            return 1
    soup = BeautifulSoup(html, "lxml")
    pages = soup.select("nav a")
    nums = []
    for a in pages:
        try:
            nums.append(int(a.text.strip()))
        except:
            pass
    return max(nums) if nums else 1

def extract_json_blocks_after_key(html: str, key: str) -> List[str]:
    results = []
    start = 0
    while True:
        idx = html.find(key, start)
        if idx == -1:
            break
        j = idx + len(key)
        while j < len(html) and html[j] not in '[{':
            j += 1
        if j >= len(html):
            start = idx + 1
            continue
        open_ch = html[j]
        close_ch = ']' if open_ch == '[' else '}'
        depth, k = 0, j
        while k < len(html):
            if html[k] == open_ch:
                depth += 1
            elif html[k] == close_ch:
                depth -= 1
                if depth == 0:
                    results.append(html[j:k+1])
                    start = k + 1
                    break
            k += 1
    return results

def safe_json_load(s: str):
    try:
        return json.loads(s)
    except Exception:
        s2 = re.sub(r",\s*([\]}])", r"\1", s)
        try:
            return json.loads(s2)
        except Exception:
            return None

def normalize_review(obj: dict) -> dict:
    v = {}
    v['title'] = obj.get('title') or obj.get('value',{}).get('title')
    v['text']  = obj.get('text') or obj.get('value',{}).get('text')
    v['rating'] = obj.get('rating') or obj.get('value',{}).get('rating')
    v['author'] = obj.get('author') or obj.get('value',{}).get('author')
    v['created'] = obj.get('created') or obj.get('value',{}).get('created')
    verified = obj.get('value',{}).get('certifiedBuyer') if isinstance(obj.get('value'), dict) else None
    v['verified'] = bool(verified) if verified is not None else None
    v['helpful'] = obj.get('helpfulCount') or obj.get('value',{}).get('helpfulCount')
    v['review_id'] = obj.get('id') or obj.get('value',{}).get('id')
    loc = obj.get('value',{}).get('location') or obj.get('location')
    v['city'] = loc.get('city') if isinstance(loc, dict) else None
    v['state'] = loc.get('state') if isinstance(loc, dict) else None
    return v

def extract_reviews_from_html(html: str) -> List[dict]:
    reviews = []
    blocks = extract_json_blocks_after_key(html, '"renderableComponents"')
    for b in blocks:
        parsed = safe_json_load(b)
        if not parsed:
            continue
        items = parsed if isinstance(parsed, list) else parsed.get('renderableComponents') or [parsed]
        for it in items:
            if not isinstance(it, dict):
                continue
            candidates = [it]
            if isinstance(it.get('value'), dict):
                candidates.append(it['value'])
            vv = it.get('value',{}).get('value')
            if isinstance(vv, dict):
                candidates.append(vv)
            for c in candidates:
                if not isinstance(c, dict):
                    continue
                nr = normalize_review(c)
                if any([nr.get('author'), nr.get('text'), nr.get('rating'), nr.get('review_id')]):
                    reviews.append(nr)
    if not reviews:
        soup = BeautifulSoup(html, "lxml")
        rev_elems = soup.select("div._16PBlm") or soup.select("div._27M-vq")
        for r in rev_elems:
            text = r.select_one("div.t-ZTKy") or r.select_one("div._2-N8zT")
            title = r.select_one("p._2xg6ul") or r.select_one("p._2xg6ul")
            rating = r.select_one("div._3LWZlK")
            author = r.select_one("p._2sc7ZR") or r.select_one("p._2sc7ZR")
            review = {"title": (title.text.strip() if title else None),
                      "text": (text.text.strip() if text else None),
                      "rating": (rating.text.strip() if rating else None),
                      "author": (author.text.strip() if author else None)}
            if review["text"] or review["title"]:
                reviews.append(review)
    return reviews

def scrape_flipkart_reviews(base_url: str, max_reviews: int = MAX_REVIEWS, max_pages: int = MAX_PAGES) -> List[dict]:
    review_url = to_review_url(base_url)
    resp = requests.get(review_url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    total_pages = extract_total_pages(resp.text)
    pages_to_scan = min(total_pages, max_pages)
    unique = []
    seen = set()

    page_reviews = extract_reviews_from_html(resp.text)
    for r in page_reviews:
        key = r.get('review_id') or (r.get('author'), (r.get('text') or '')[:80], r.get('rating'))
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
        if len(unique) >= max_reviews:
            break

    if len(unique) < max_reviews and pages_to_scan >= 2:
        for p in range(2, pages_to_scan + 1):
            url = f"{review_url}?page={p}"
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code != 200:
                continue
            page_reviews = extract_reviews_from_html(r.text)
            for rev in page_reviews:
                key = rev.get('review_id') or (rev.get('author'), (rev.get('text') or '')[:80], rev.get('rating'))
                if key in seen:
                    continue
                seen.add(key)
                unique.append(rev)
                if len(unique) >= max_reviews:
                    break
            if len(unique) >= max_reviews:
                break

    return unique[:max_reviews]


# ---------------- NEW: product scraping for description + specs ----------------
def scrape_product_page(product_url: str) -> Dict[str, any]:
    """
    Fetch product page and extract:
     - title
     - features: list of {heading, text, img}
     - specs: dict of groups -> {key: value}
    """
    resp = requests.get(product_url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    title_tag = soup.find("span", {"class": "B_NuCI"}) or soup.find("span", {"class": "_35KyD6"})
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Feature blocks (Product Description)
    features = []
    feature_blocks = soup.find_all("div", class_=re.compile(r"pqHCzB|CB-A\+e"))
    for fb in feature_blocks:
        heading = fb.find("div", class_=re.compile(r"_9GQWrZ"))
        heading_text = heading.get_text(strip=True) if heading else ""
        para = fb.find("div", class_=re.compile(r"AoD2-N"))
        para_text = para.get_text(" ", strip=True) if para else ""
        img = fb.find("img")
        img_src = img.get("src") if img else ""
        if heading_text or para_text or img_src:
            features.append({"heading": heading_text, "text": para_text, "img": img_src})

    # Specs parsing
    specs = {}
    spec_header = soup.find(lambda tag: tag.name in ("div","h2","span") and "Specifications" in tag.get_text())
    if spec_header:
        spec_parent = spec_header.find_parent()
        if spec_parent:
            for table in spec_parent.find_all("table"):
                prev = table.find_previous(lambda t: t.name in ("div","h3","span") and len(t.get_text(strip=True))>0)
                group_name = prev.get_text(strip=True) if prev else "General"
                rows = table.find_all("tr")
                group_dict = specs.setdefault(group_name, {})
                for tr in rows:
                    tds = tr.find_all("td")
                    if len(tds) >= 2:
                        key = tds[0].get_text(" ", strip=True)
                        val = tds[1].get_text(" ", strip=True)
                        if key:
                            group_dict[key] = val

    product = {"product_url": product_url, "title": title, "features": features, "specs": specs}
    return product


# ---------------- Sentiment analysis ----------------
def analyze_sentiment(reviews: List[dict]) -> Tuple[List[dict], dict]:
    if not reviews:
        return [], {"positive":0,"negative":0,"neutral":0,"total":0}
    nlp = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, device=-1)
    per = []
    counts = {"POSITIVE":0,"NEGATIVE":0}
    for r in reviews:
        text = ((r.get("title") or "") + " " + (r.get("text") or "")).strip()
        if not text:
            continue
        res = nlp(text[:512])[0]
        label = res["label"]
        score = float(res.get("score",0.0))
        counts[label] = counts.get(label,0) + 1
        per.append({"author": r.get("author"), "rating": r.get("rating"), "text": r.get("text"), "label": label, "score": score})
    total = counts.get("POSITIVE",0)+counts.get("NEGATIVE",0)
    agg = {"positive": counts.get("POSITIVE",0), "negative": counts.get("NEGATIVE",0), "total": total}
    return per, agg


# ---------------- Build embeddings + FAISS index ----------------
def build_rag_index_from_reviews(reviews: List[dict], product: Optional[dict] = None, embed_model_name: str = EMBED_MODEL_NAME) -> Tuple[FAISS, List[Document]]:
    docs = []
    # add product description as a single doc (if provided) so it can be retrieved
    if product:
        # flatten product into text block
        prod_parts = []
        if product.get("title"):
            prod_parts.append(f"Product Title: {product['title']}")
        if product.get("features"):
            prod_parts.append("Product Features:")
            for f in product["features"]:
                h = f.get("heading") or ""
                t = f.get("text") or ""
                prod_parts.append(f"- {h}: {t}")
        if product.get("specs"):
            prod_parts.append("Product Specifications:")
            for group, mapping in product["specs"].items():
                prod_parts.append(f"{group}:")
                for k, v in mapping.items():
                    prod_parts.append(f"  - {k}: {v}")
        prod_text = "\n".join(prod_parts)
        docs.append(Document(page_content=prod_text, metadata={"source":"product_description"}))

    for i, r in enumerate(reviews):
        content = (r.get("title") or "") + "\n\n" + (r.get("text") or "")
        metadata = {"source": f"review_{i}", "author": r.get("author"), "rating": r.get("rating")}
        docs.append(Document(page_content=content, metadata=metadata))

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
    index = FAISS.from_documents(chunks, embeddings)
    return index, chunks


# ---------------- RAG: retrieval + generate ----------------
def rag_answer(question: str, index: FAISS, chunks: List[Document], product: Optional[dict], ollama_client: OllamaClient, top_k: int = TOP_K) -> dict:
    retriever = index.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(question)
    ctx_parts = []
    for i, d in enumerate(docs):
        src = d.metadata.get("source", f"chunk_{i}")
        header = f"[source: {src}]"
        ctx_parts.append(f"{header}\n{d.page_content}")
    context = "\n\n---\n\n".join(ctx_parts) if ctx_parts else ""

    # include explicit product description block if available
    product_block = ""
    if product:
        # concise product summary for the model
        ptitle = product.get("title","")
        pfeat_lines = []
        for f in product.get("features", [])[:10]:
            if f.get("heading") or f.get("text"):
                pfeat_lines.append(f"{f.get('heading','').strip()} - {f.get('text','').strip()}")
        pspec_lines = []
        # include a few specs (top-level keys)
        for grp, mp in list(product.get("specs", {}).items())[:6]:
            # flatten a couple of specs from each group
            for k,v in list(mp.items())[:3]:
                pspec_lines.append(f"{grp}: {k} = {v}")
        product_block = "Product description:\n"
        if ptitle:
            product_block += f"Title: {ptitle}\n"
        if pfeat_lines:
            product_block += "Features:\n" + "\n".join(f"- {l}" for l in pfeat_lines) + "\n"
        if pspec_lines:
            product_block += "Specifications (sample):\n" + "\n".join(f"- {l}" for l in pspec_lines) + "\n"

    # improved assistant prompt (friendly assistant persona, cite sources, avoid hallucination)
    prompt = f"""You are a helpful, friendly assistant that answers user questions about a Flipkart product using the available context.
You speak like a human assistant — concise, polite, and helpful. For casual greetings (Hi/Hello) respond naturally (e.g. "Hi! How can I help?").
Always prefer to use information from the provided Product description and retrieved review passages. Quote or reference sources when you use them (use the source tags like [source: review_3] or [source: product_description]).
If the answer is not present in the context, reply: "I don't know." Do not hallucinate facts.

Product context (if any):
{product_block}

Retrieved passages:
{context}

Question: {question}

Answer in a concise helpful way. If you include steps or suggestions, number them. End with a short one-line summary.
"""
    answer = ollama_client.generate(prompt, temperature=0.0, max_tokens=512)
    return {"answer": answer, "sources": [d.metadata.get("source","unknown") for d in docs], "retrieved": len(docs)}


# ---------------- Gradio UI ----------------
ollama_client = OllamaClient()

# session storage (now includes product)
SESSION = {"index": None, "chunks": None, "reviews": None, "sentiment": None, "product": None, "chat": []}

def handle_scrape_and_analyze(url: str):
    try:
        reviews = scrape_flipkart_reviews(url, max_reviews=MAX_REVIEWS, max_pages=MAX_PAGES)
    except Exception as e:
        status_str = f"Scrape failed: {e}"
        return status_str, [], "**No results**"
    if not reviews:
        status_str = "No reviews found (maybe URL invalid or reviews unavailable)."
        return status_str, [], "**No results**"

    # fetch product page (optional) and attach product info to session
    try:
        product_url = to_product_url(url)
        product = scrape_product_page(product_url)
        SESSION["product"] = product
    except Exception as e:
        # if product scraping fails, continue without product
        product = None
        SESSION["product"] = None
        print("Product scraping failed:", e)

    per, agg = analyze_sentiment(reviews)
    try:
        index, chunks = build_rag_index_from_reviews(reviews, product=product)
    except Exception as e:
        status_str = f"Indexing failed: {e}"
        return status_str, [], "**Indexing failed**"

    SESSION["index"] = index
    SESSION["chunks"] = chunks
    SESSION["reviews"] = reviews
    SESSION["sentiment"] = {"per": per, "agg": agg}
    status = f"Scraped {len(reviews)} reviews. Sentiment: {agg['positive']} positive, {agg['negative']} negative (total {agg['total']})"
    rows = []
    for p in per:
        rows.append([p.get("author"), p.get("rating"), p.get("label"), round(p.get("score",0),3), (p.get("text") or "")[:300]])
    agg_md = f"**Aggregate** — positive: {agg['positive']}  negative: {agg['negative']}  total: {agg['total']}"
    return status, rows, agg_md

def handle_rag_query(question: str):
    if SESSION["index"] is None:
        return SESSION["chat"], "Index not ready. Please paste a Flipkart URL and press Scrape first."
    SESSION["chat"].append({"role":"user","content": question})
    try:
        resp = rag_answer(question, SESSION["index"], SESSION["chunks"], SESSION.get("product"), ollama_client, top_k=TOP_K)
    except Exception as e:
        SESSION["chat"].append({"role":"assistant","content": f"Error: {e}"})
        return SESSION["chat"], ""
    answer = resp.get("answer") or ""
    sources = resp.get("sources", [])
    bot_msg = f"{answer}\n\nSources: {', '.join(sources)}"
    SESSION["chat"].append({"role":"assistant","content": bot_msg})
    return SESSION["chat"], ""

with gr.Blocks(title="Flipkart Review RAG + Sentiment") as demo:
    gr.Markdown("# Flipkart Review Assistant")
    gr.Markdown("Paste a Flipkart product or review URL, press **Scrape & Analyze**, then ask questions grounded in those reviews and product description.")
    with gr.Row():
        url_in = gr.Textbox(label="Flipkart product / review URL", placeholder="https://www.flipkart.com/....", lines=1)
        btn = gr.Button("Scrape & Analyze")
    status = gr.Textbox(label="Status", interactive=False)
    with gr.Tab("Sentiment"):
        sentiment_output = gr.Dataframe(headers=["author","rating","label","score","text"], label="Per-review sentiment (sample)")
        agg_text = gr.Markdown("")
    with gr.Tab("Chat (RAG)"):
        chatbox = gr.Chatbot(type="messages")
        user_input = gr.Textbox(label="Ask a question about the reviews/product", placeholder="e.g. What do people say about battery life?")
        send_btn = gr.Button("Ask")

    btn.click(fn=handle_scrape_and_analyze, inputs=[url_in], outputs=[status, sentiment_output, agg_text])
    send_btn.click(fn=handle_rag_query, inputs=[user_input], outputs=[chatbox, user_input])

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)
