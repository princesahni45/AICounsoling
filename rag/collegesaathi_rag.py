"""CollegeSaathi RAG pipeline (AKTU-affiliated UP colleges only).

Drop-in, framework-agnostic. Optional deps:
  chromadb, sentence-transformers  -> used automatically if installed
Fallbacks (pure python) are used otherwise and are reported by status().
"""
import hashlib, io, json, math, os, re, time, zlib
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

DATA_DIR = os.environ.get("RAG_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
REGISTRY_PATH = os.path.join(DATA_DIR, "aktu_colleges.json")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
UA = {"User-Agent": "CollegeSaathiBot/1.0 (student project; respects robots by manual allowlist)"}

TOPIC_KEYWORDS = {
    "fees": ["fee", "fees", "tuition", "per year", "per semester", "caution money", "rs.", "inr", "₹"],
    "hostel": ["hostel", "mess", "warden", "boarding", "accommodation", "room"],
    "placement": ["placement", "package", "lpa", "recruiter", "ctc", "campus drive", "offers"],
    "courses": ["b.tech", "btech", "branch", "intake", "course", "program", "m.tech", "mba", "mca", "seats"],
    "admission": ["admission", "uptac", "jee", "counselling", "eligibility", "cutoff", "rank", "apply"],
    "facilities": ["library", "lab", "sports", "wifi", "canteen", "transport", "bus", "campus"],
}
STOP = set("the a an of to in and or for is are what which how do does at on with my me i it this that".split())


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def _save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------- registry
def load_registry():
    return {c["id"]: c for c in _load(REGISTRY_PATH, [])}


def verify_source(college, url):
    """Ingest only if affiliation is verified (with proof URL) and URL host belongs to the college."""
    if not college.get("affiliation_verified") or not college.get("aktu_affiliation_source"):
        raise PermissionError(f"{college['id']}: AKTU affiliation not verified in registry; refusing to ingest")
    host = (urlparse(url).hostname or "").lower()
    if not any(host == d or host.endswith("." + d) for d in college.get("domains", [])):
        raise PermissionError(f"{url}: host not in registered domains {college.get('domains')}")


# ---------------------------------------------------------------- fetch/extract
class ExtractionError(Exception):
    pass


def fetch(url, timeout=20):
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    ctype = r.headers.get("content-type", "").lower()
    if "pdf" in ctype or url.lower().split("?")[0].endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(r.content))
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
        if len(text.strip()) < 40:
            raise ExtractionError("PDF has no usable text layer (scanned? OCR not implemented)")
        return clean_text(text), "pdf"
    soup = BeautifulSoup(r.text, "html.parser")
    for tr in soup.find_all("tr"):  # keep table rows (fee tables) on one line
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        tr.replace_with(soup.new_string(" | ".join(cells) + "\n"))
    for t in soup(["script", "style", "noscript", "nav", "footer", "form", "svg"]):
        t.decompose()
    text = clean_text(soup.get_text("\n"))
    if len(text) < 40:
        raise ExtractionError("HTML page has almost no text (JS-rendered? not supported)")
    return text, "html"


def clean_text(text):
    out, prev = [], None
    for line in text.replace("\xa0", " ").splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if len(line) < 2 or line == prev:
            continue
        out.append(line)
        prev = line
    return "\n".join(out)


# ---------------------------------------------------------------- chunking
def chunk_text(text, max_chars=900, overlap=150):
    lines = []
    for ln in text.split("\n"):
        while len(ln) > max_chars:
            lines.append(ln[:max_chars]); ln = ln[max_chars:]
        lines.append(ln)
    chunks, cur = [], []
    for ln in lines:
        if cur and sum(len(x) + 1 for x in cur) + len(ln) > max_chars:
            chunks.append("\n".join(cur))
            tail, n = [], 0
            for x in reversed(cur):
                n += len(x) + 1
                if n > overlap:
                    break
                tail.insert(0, x)
            cur = tail
        cur.append(ln)
    if cur:
        chunks.append("\n".join(cur))
    return [c for c in chunks if len(c) > 30]


def classify_topic(text):
    low = text.lower()
    scores = {t: sum(low.count(k) for k in kws) for t, kws in TOPIC_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def infer_topic(question):
    low = question.lower()
    scores = {t: sum(k in low for k in kws) for t, kws in TOPIC_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


# ---------------------------------------------------------------- embeddings
class HashingEmbedder:
    """Pure-python fallback (lexical, NOT semantic). Used only if sentence-transformers is missing."""
    name, dim = "hashing-fallback", 512

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"\w+", t.lower()):
                if w in STOP:
                    continue
                h = zlib.crc32(w.encode())
                v[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class STEmbedder:
    def __init__(self, model):
        from sentence_transformers import SentenceTransformer
        self.m, self.name = SentenceTransformer(model), model

    def embed(self, texts):
        return self.m.encode(texts, normalize_embeddings=True).tolist()


def get_embedder():
    try:
        return STEmbedder(EMBED_MODEL)
    except Exception:
        return HashingEmbedder()


# ---------------------------------------------------------------- vector store
def _match(meta, where):
    if not where:
        return True
    for k, v in where.items():
        if k == "$and":
            if not all(_match(meta, w) for w in v):
                return False
        elif isinstance(v, dict) and "$in" in v:
            if meta.get(k) not in v["$in"]:
                return False
        elif meta.get(k) != v:
            return False
    return True


class JsonStore:
    """Fallback store with the same interface we use from Chroma."""
    name = "json-fallback"

    def __init__(self, path):
        self.path, self.rows = path, _load(path, {})

    def upsert(self, ids, embeddings, documents, metadatas):
        for i, e, d, m in zip(ids, embeddings, documents, metadatas):
            self.rows[i] = {"e": e, "d": d, "m": m}
        _save(self.path, self.rows)

    def delete(self, where):
        self.rows = {k: v for k, v in self.rows.items() if not _match(v["m"], where)}
        _save(self.path, self.rows)

    def query(self, emb, n, where=None):
        sc = [(sum(a * b for a, b in zip(emb, r["e"])), r) for r in self.rows.values() if _match(r["m"], where)]
        sc.sort(key=lambda x: -x[0])
        return [(s, r["d"], r["m"]) for s, r in sc[:n]]

    def count(self):
        return len(self.rows)


class ChromaStore:
    name = "chromadb"

    def __init__(self, path):
        import chromadb
        self.c = chromadb.PersistentClient(path=path).get_or_create_collection(
            "aktu_colleges", metadata={"hnsw:space": "cosine"})

    def upsert(self, ids, embeddings, documents, metadatas):
        self.c.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def delete(self, where):
        self.c.delete(where=where)

    def query(self, emb, n, where=None):
        cnt = self.c.count()
        if cnt == 0:
            return []
        r = self.c.query(query_embeddings=[emb], n_results=min(n, cnt), where=where or None)
        return [(1 - d, doc, m) for d, doc, m in zip(r["distances"][0], r["documents"][0], r["metadatas"][0])]

    def count(self):
        return self.c.count()


def get_store():
    try:
        return ChromaStore(os.path.join(DATA_DIR, "chroma"))
    except Exception:
        return JsonStore(os.path.join(DATA_DIR, "vectors.json"))


# ---------------------------------------------------------------- pipeline
class RAG:
    def __init__(self, embedder=None, store=None):
        self.embedder = embedder or get_embedder()
        self.store = store or get_store()

    def status(self):
        man = _load(MANIFEST_PATH, {})
        return {"embedder": self.embedder.name, "store": self.store.name, "chunks": self.store.count(),
                "sources": len(man), "failing_sources": [u for u, m in man.items() if m.get("status") != "ok"]}

    # ---- ingestion / updates (Phases 1, 2, 5)
    def ingest_url(self, college_id, url, academic_year):
        reg = load_registry()
        if college_id not in reg:
            raise PermissionError(f"{college_id} not in AKTU registry")
        college = reg[college_id]
        verify_source(college, url)
        man = _load(MANIFEST_PATH, {})
        entry = man.get(url, {"college_id": college_id, "academic_year": academic_year, "n_chunks": 0})
        entry["last_checked"] = now()
        try:
            text, kind = fetch(url)
        except Exception as e:  # broken URL / extraction failure: keep old chunks, mark stale
            entry.update(status="error", error=f"{type(e).__name__}: {e}")
            man[url] = entry; _save(MANIFEST_PATH, man)
            return {"url": url, "result": "error", "error": entry["error"]}
        h = hashlib.sha256(text.encode()).hexdigest()
        if entry.get("content_hash") == h and entry.get("academic_year") == academic_year and entry.get("status") == "ok":
            man[url] = entry; _save(MANIFEST_PATH, man)
            return {"url": url, "result": "unchanged"}
        chunks = chunk_text(text)
        metas, ids, docs, seen = [], [], [], set()
        for i, c in enumerate(chunks):
            ch = hashlib.sha1((college_id + c).encode()).hexdigest()
            if ch in seen:  # exact duplicate chunk within a college
                continue
            seen.add(ch)
            ids.append(f"{college_id}:{ch}")
            docs.append(c)
            metas.append({"college_id": college_id, "college_name": college["name"], "source_url": url,
                          "academic_year": academic_year, "topic": classify_topic(c), "kind": kind,
                          "last_fetched": now(), "chunk_index": i})
        embs = self.embedder.embed([f"{college['name']} | {m['topic']}: {d}" for d, m in zip(docs, metas)])
        self.store.delete({"source_url": url})  # replace old version of this source
        self.store.upsert(ids, embs, docs, metas)
        entry.update(content_hash=h, last_fetched=now(), status="ok", error=None, n_chunks=len(ids),
                     academic_year=academic_year)
        man[url] = entry; _save(MANIFEST_PATH, man)
        return {"url": url, "result": "updated", "chunks": len(ids)}

    def refresh_all(self):
        man = _load(MANIFEST_PATH, {})
        return [self.ingest_url(m["college_id"], u, m["academic_year"]) for u, m in list(man.items())]

    # ---- retrieval (Phase 3)
    def detect_colleges(self, question):
        low, found = question.lower(), []
        for cid, c in load_registry().items():
            names = [c["name"]] + c.get("aliases", [])
            if any(re.search(r"\b" + re.escape(n.lower()) + r"\b", low) for n in names):
                found.append(cid)
        return found

    def retrieve(self, question, college_ids, topic=None, academic_year=None, k=4):
        qe = self.embedder.embed([question])[0]
        hits = []
        for cid in college_ids:  # k per college so comparisons get both sides
            conds = [{"college_id": cid}]
            if academic_year:
                conds.append({"academic_year": academic_year})
            base = {"$and": conds} if len(conds) > 1 else conds[0]
            res = []
            if topic:
                res = self.store.query(qe, k, {"$and": conds + [{"topic": topic}]})
            if not res:
                res = self.store.query(qe, k, base)
            hits += res
        return hits

    # ---- answer (Phase 4)
    def answer(self, question, llm=None, session_college_ids=None, academic_year=None):
        colleges = self.detect_colleges(question) or list(session_college_ids or [])
        if not colleges:
            return {"route": "need_college", "answer": "Which AKTU college are you asking about? "
                    "I only answer from sources I have ingested for specific colleges.", "sources": []}
        hits = self.retrieve(question, colleges, infer_topic(question), academic_year)
        if not hits:
            return {"route": "no_data", "sources": [], "answer": "I don't have ingested information for that "
                    "college yet, so I can't answer this reliably. Please check the college's official website."}
        sources, seen = [], {}
        for _, _, m in hits:
            if m["source_url"] not in seen:
                seen[m["source_url"]] = len(seen) + 1
                sources.append({"n": seen[m["source_url"]], "college": m["college_name"], "url": m["source_url"],
                                "academic_year": m["academic_year"], "last_fetched": m["last_fetched"]})
        ctx = "\n\n".join(f"[{seen[m['source_url']]}] ({m['college_name']}, {m['topic']}, "
                          f"AY {m['academic_year']})\n{d}" for _, d, m in hits)
        if llm:
            system = ("You are CollegeSaathi, a friendly counsellor for AKTU-affiliated colleges in Uttar Pradesh. "
                      "Answer ONLY from the CONTEXT. Never invent or estimate fees, cutoffs, ranks or placement "
                      "numbers. If the context lacks the answer, say it is not available in the ingested sources. "
                      "Cite sources as [n] and mention the academic year of any figure you use.")
            text = llm(system, f"CONTEXT:\n{ctx}\n\nQUESTION: {question}")
        else:  # extractive fallback: show the retrieved passages, no generation
            text = "Here is what the ingested sources say:\n\n" + ctx
        refs = "\n".join(f"[{s['n']}] {s['college']} - {s['url']} (academic year {s['academic_year']}, "
                         f"fetched {s['last_fetched']})" for s in sources)
        return {"route": "rag", "answer": f"{text}\n\nSources:\n{refs}\n(Fetched copies; verify on the official site "
                "before deciding.)", "sources": sources}


RANK_RE = re.compile(r"\b(rank|jee|category|obc|sc|st|ews|general)\b.*\b(suit|college|get|chance|recommend)|"
                     r"\b(suit|recommend|chance)\b.*\b(rank|jee)", re.I)


def route(question):
    """Recommendation questions go to the existing cutoff-based recommender, never to the LLM."""
    return "recommend" if RANK_RE.search(question) else "rag"


if __name__ == "__main__":
    import sys
    r = RAG()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "ingest":
        print(r.ingest_url(sys.argv[2], sys.argv[3], sys.argv[4]))
    elif cmd == "refresh":
        print(json.dumps(r.refresh_all(), indent=2))
    elif cmd == "ask":
        print(r.answer(" ".join(sys.argv[2:]))["answer"])
    else:
        print(json.dumps(r.status(), indent=2))
