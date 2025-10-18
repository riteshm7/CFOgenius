from flask import Flask, render_template, request
from kronoslabs import KronosLabs
import os, json, glob, re
from werkzeug.utils import secure_filename

# ---------- CONFIG ----------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "small", "train")
MAX_SHARDS = int(os.environ.get("MAX_SHARDS", 2))

REPORT_KEYS = [
    "section_1","section_1A","section_1B","section_2","section_3","section_4",
    "section_5","section_6","section_7","section_7A","section_8","section_9",
    "section_9A","section_9B","section_10","section_11","section_12","section_13",
    "section_14","section_15"
]
TARGET_SECTIONS_PRIMARY = {"section_7", "section_7A"}   # MD&A
TARGET_SECTIONS_SECONDARY = {"section_1A"}               # Risk factors

# File uploads
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Kronos key
KRONOS_API_KEY = os.environ.get("KRONOS_API_KEY", "kl_786d900daf441cb13ab394be57191ae0f7e28a213316d4a4f7bd562f753341e5")
client = KronosLabs(api_key=KRONOS_API_KEY)

# ---------- HELPERS ----------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def _clean_text(t: str) -> str:
    # Normalize ligatures, bullets, quotes, dashes, symbols and whitespace
    replacements = {
        "\u2022": " • ",  # bullet
        "\u00A0": " ",    # nbsp
        "ﬁ": "fi",
        "ﬂ": "fl",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "•": " • ",
        "®": "",
        "™": ""
    }
    for k, v in replacements.items():
        t = t.replace(k, v)

    # Join hyphenated line breaks: re-\nport -> report
    t = re.sub(r"-\s*\n\s*", "", t)

    # Replace newlines with spaces, collapse multiple spaces
    t = re.sub(r"\s*\n\s*", " ", t)
    t = re.sub(r"\s{2,}", " ", t)

    return t.strip()

def _strip_headers_and_footers(t: str) -> str:
    # Remove common SEC boilerplate / headings / page refs
    patterns = [
        r"UNITED STATES SECURITIES AND EXCHANGE COMMISSION.*?$",
        r"Form\s+10\-K",
        r"Table of Contents",
        r"PART\s+[I|II|III|IV]",
        r"Item\s+\d+[\.\: ]",
        r"Page\s+\d+",
    ]
    for p in patterns:
        t = re.sub(p, " ", t, flags=re.IGNORECASE | re.MULTILINE)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()

def load_sentences_from_jsonl(data_dir: str, max_shards: int = 2):
    sentences = []
    shard_paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.jsonl")))
    if not shard_paths:
        print(f"⚠️  No shards found in {data_dir}. Put shard_*.jsonl there.")
        return sentences

    loaded_files = 0
    for path in shard_paths[:max_shards]:
        loaded_files += 1
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                firm = json.loads(line)
                firm_name = firm.get("name", "")
                tickers = firm.get("tickers", []) or []
                for filing in firm.get("filings", []):
                    report = filing.get("report", {})
                    report_date = filing.get("reportDate", "")
                    year = report_date.split("-")[0] if report_date else ""
                    for sec in REPORT_KEYS:
                        sec_content = report.get(sec)
                        if not sec_content:
                            continue
                        for s in sec_content:
                            if s and s.strip():
                                txt = _clean_text(s)
                                if txt:
                                    sentences.append({
                                        "text": txt,
                                        "company": firm_name,
                                        "tickers": tickers,
                                        "section": sec,
                                        "year": year
                                    })
    print(f"✅ Loaded {len(sentences)} sentences from {loaded_files} shard(s).")
    return sentences

TRAIN_SENTENCES = load_sentences_from_jsonl(DATA_DIR, MAX_SHARDS)

def select_company_sentences(query: str, sentences, max_total=40):
    q = query.lower()

    # Strict company/ticker match
    company_hits = [
        s for s in sentences
        if (s["company"] and q in s["company"].lower())
        or any(q == t.lower() for t in s["tickers"])
    ]

    # Fallback: keyword match
    keyword_hits = [s for s in sentences if q in s["text"].lower()]

    hits = company_hits if company_hits else keyword_hits
    if not hits:
        return []

    primary = [s for s in hits if s["section"] in TARGET_SECTIONS_PRIMARY]
    secondary = [s for s in hits if s["section"] in TARGET_SECTIONS_SECONDARY]
    others = [s for s in hits if s["section"] not in (TARGET_SECTIONS_PRIMARY | TARGET_SECTIONS_SECONDARY)]
    ordered = primary + secondary + others

    seen = set()
    kept = []
    for s in ordered:
        t = s["text"]
        if t in seen:
            continue
        seen.add(t)
        kept.append(s)
        if len(kept) >= max_total:
            break
    return kept

def build_advice_prompt(company_query: str, snippets: list) -> str:
    lines = []
    for i, s in enumerate(snippets[:40], 1):
        tag = f"{s.get('company','Uploaded Doc')} | {','.join(s.get('tickers',[]))} | {s.get('section','?')} | {s.get('year','?')}"
        lines.append(f"{i}. [{tag}] {s['text']}")
    context = "\n".join(lines)

    prompt = f"""
You are a seasoned CFO/COO advisor. Based ONLY on the excerpts below, provide practical, prioritized advice to improve the business.

Constraints:
- Be specific and realistic. No generic fluff.
- Tie each recommendation to what the excerpts imply (ops, finance, product, sales/GTM, compliance).
- Organize output: 1) Quick Wins (0–90 days), 2) Medium-Term (3–12 months), 3) Long-Term (12–24 months).
- For each item include: Rationale (from text), Expected Impact, Owner, First Action Step.
- If data is weak/boilerplate, say what additional data is needed.
- Ignore table-of-contents, exhibit lists, and filing cover boilerplate.

Query: {company_query}

EXCERPTS:
{context}

Return the advisory plan in concise bullet points with short headers.
""".strip()
    return prompt

# ---------- FILE TEXT EXTRACTORS ----------
def extract_text_from_pdf(path: str) -> str:
    # Try pdfplumber first (often cleaner layout)
    try:
        import pdfplumber
        parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                parts.append(txt)
        text = "\n".join(parts)
        if text.strip():
            text = _strip_headers_and_footers(_clean_text(text))
            return text
    except Exception:
        pass

    # Fallback: pdfminer.six
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        text = pdfminer_extract(path) or ""
        text = _strip_headers_and_footers(_clean_text(text))
        return text
    except Exception as e:
        return f"ERROR: Failed to read PDF ({e})"

def extract_text_from_docx(path: str) -> str:
    try:
        import docx  # python-docx
    except ImportError:
        return "ERROR: python-docx not installed. Run: pip3 install python-docx"
    try:
        doc = docx.Document(path)
        text = "\n".join(p.text for p in doc.paragraphs)
        return _strip_headers_and_footers(_clean_text(text))
    except Exception as e:
        return f"ERROR: Failed to read DOCX ({e})"

def extract_text_from_txt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return _strip_headers_and_footers(_clean_text(text))
    except Exception as e:
        return f"ERROR: Failed to read TXT ({e})"

def extract_text_from_file(path: str) -> str:
    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(path)
    if ext == "docx":
        return extract_text_from_docx(path)
    if ext == "txt":
        return extract_text_from_txt(path)
    return "ERROR: Unsupported file type."

def split_into_snippets(text: str, max_chars=800, max_snippets=40):
    """
    Split cleaned text into sentence-ish chunks and pack into ~max_chars snippets.
    Skips tiny noise fragments.
    """
    if text.startswith("ERROR:"):
        return []

    text = _strip_headers_and_footers(_clean_text(text))

    # Rough sentence split: end punctuation or semicolons/newlines
    parts = re.split(r'(?<=[\.\?\!;])\s+', text)
    parts = [p.strip() for p in parts if p.strip()]

    snippets = []
    current = ""
    for s in parts:
        # Skip super-short rubbish like "1." / "2." / "Item 1A"
        if len(s) < 20 and re.match(r'^[\-\dA-Za-z\.\(\) ]{1,10}$', s):
            continue

        candidate = (current + " " + s).strip() if current else s
        if len(candidate) > max_chars:
            if current:
                snippets.append({"text": current, "company": "Uploaded Doc", "tickers": [], "section": "uploaded", "year": ""})
            current = s
        else:
            current = candidate

        if len(snippets) >= max_snippets:
            break

    if current and len(snippets) < max_snippets:
        snippets.append({"text": current, "company": "Uploaded Doc", "tickers": [], "section": "uploaded", "year": ""})

    return snippets

# ---------- FLASK ----------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25MB cap

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    upload_result = None
    error = None
    upload_error = None

    # Form 1: dataset search (company/ticker/keyword)
    if request.method == "POST" and "input_value" in request.form:
        query = request.form.get("input_value", "").strip()
        if not TRAIN_SENTENCES:
            error = "No local data loaded. Add JSONL shards to data/small/train/."
        elif not query:
            error = "Enter a company, ticker, or keyword (e.g., Apple, AAPL, revenue)."
        else:
            snippets = select_company_sentences(query, TRAIN_SENTENCES, max_total=40)
            if not snippets:
                error = "No matching excerpts found in loaded shards. Download more shards or try another query."
            else:
                prompt = build_advice_prompt(query, snippets)
                try:
                    resp = client.chat.completions.create(
                        prompt=prompt,
                        model="hyperion",
                        temperature=0.4,
                        is_stream=False
                    )
                    text = (resp.choices[0].message.content or "")
                    result = text[:5000]
                except Exception as e:
                    error = f"Kronos API error: {e}"

    # Form 2: file upload
    if request.method == "POST" and "report_file" in request.files:
        file = request.files.get("report_file")
        query2 = request.form.get("upload_query", "").strip()  # optional context/topic
        if not file or file.filename == "":
            upload_error = "Please choose a PDF, DOCX, or TXT file."
        elif not allowed_file(file.filename):
            upload_error = "Unsupported file type. Allowed: PDF, DOCX, TXT."
        else:
            fname = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
            file.save(save_path)

            text = extract_text_from_file(save_path)
            snippets = split_into_snippets(text, max_chars=800, max_snippets=40)
            if not snippets:
                upload_error = "Could not extract text from the file. Make sure it contains selectable text."
            else:
                prompt = build_advice_prompt(query2 or "Uploaded Business Report", snippets)
                try:
                    resp = client.chat.completions.create(
                        prompt=prompt,
                        model="hyperion",
                        temperature=0.4,
                        is_stream=False
                    )
                    text = (resp.choices[0].message.content or "")
                    upload_result = text[:5000]
                except Exception as e:
                    upload_error = f"Kronos API error: {e}"

    return render_template(
        "index.html",
        result=result,
        error=error,
        upload_result=upload_result,
        upload_error=upload_error
    )

@app.route("/health")
def health():
    return {"ok": True, "sentences": len(TRAIN_SENTENCES)}, 200

if __name__ == "__main__":
    app.run(debug=True)