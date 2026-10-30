"""Offline pipeline test. Uses a local HTTP server with FIXTURE pages (fake numbers, test only)."""
import json, os, tempfile, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

os.environ["RAG_DATA_DIR"] = tempfile.mkdtemp()
import collegesaathi_rag as R

PAGES = {
    "/a/fees": "<html><body><nav>menu</nav><h1>Alpha Institute Fees</h1><table>"
               "<tr><th>Course</th><th>Fee per year</th></tr><tr><td>B.Tech CSE</td><td>Rs. 11111 FIXTURE</td></tr></table>"
               "<p>Hostel: separate hostels for boys and girls with mess, hostel fee Rs. 22222 FIXTURE per year.</p></body></html>",
    "/b/fees": "<html><body><h1>Beta College Fees</h1><table><tr><th>Course</th><th>Fee per year</th></tr>"
               "<tr><td>B.Tech CSE</td><td>Rs. 33333 FIXTURE</td></tr></table>"
               "<p>Placement: highest package 99 LPA FIXTURE, recruiters visit campus.</p></body></html>",
}


def make_pdf(text):
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"]
    stream = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET"
    objs += [f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream", "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out)); out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    x = len(out)
    out += f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offs).encode()
    out += f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{x}\n%%EOF".encode()
    return out


PDF = make_pdf("Alpha Institute admission brochure FIXTURE: admission through UPTAC counselling based on JEE Main rank, eligibility PCM.")


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/a/brochure.pdf":
            self.send_response(200); self.send_header("Content-Type", "application/pdf"); self.end_headers(); self.wfile.write(PDF)
        elif self.path in PAGES:
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers()
            self.wfile.write(PAGES[self.path].encode())
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, *a): pass


srv = HTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
B = f"http://127.0.0.1:{srv.server_port}"

reg = [
    {"id": "alpha", "name": "Alpha Institute (FIXTURE)", "aliases": ["alpha"], "domains": ["127.0.0.1"],
     "aktu_affiliation_source": "fixture", "affiliation_verified": True},
    {"id": "beta", "name": "Beta College (FIXTURE)", "aliases": ["beta"], "domains": ["127.0.0.1"],
     "aktu_affiliation_source": "fixture", "affiliation_verified": True},
    {"id": "gamma", "name": "Gamma (FIXTURE, unverified)", "aliases": ["gamma"], "domains": ["127.0.0.1"],
     "aktu_affiliation_source": "", "affiliation_verified": False},
]
R._save(R.REGISTRY_PATH, reg)

ok = 0
def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name); ok += bool(cond)
    assert cond, name

rag = R.RAG()
print("status:", rag.status())

# ingestion
check("ingest html alpha", rag.ingest_url("alpha", B + "/a/fees", "2025-26")["result"] == "updated")
check("ingest pdf alpha", rag.ingest_url("alpha", B + "/a/brochure.pdf", "2025-26")["result"] == "updated")
check("ingest html beta", rag.ingest_url("beta", B + "/b/fees", "2025-26")["result"] == "updated")
for cid, url in [("gamma", B + "/a/fees"), ("nonexistent", B + "/a/fees")]:
    try: rag.ingest_url(cid, url, "2025-26"); r = False
    except PermissionError: r = True
    check(f"refuse unverified/unknown college {cid}", r)
c = rag.store.count()
check("re-ingest unchanged -> no duplicates", rag.ingest_url("alpha", B + "/a/fees", "2025-26")["result"] == "unchanged" and rag.store.count() == c)

# retrieval isolation + citations
r = rag.answer("What is the B.Tech CSE fee at alpha?")
check("fee answer contains alpha figure", "11111" in r["answer"] and "33333" not in r["answer"])
check("citation has url + academic year", r["sources"] and r["sources"][0]["academic_year"] == "2025-26" and "http" in r["answer"])
r = rag.answer("what hostel facilities does alpha have")
check("hostel answer from alpha", "hostel" in r["answer"].lower() and "22222" in r["answer"])
r = rag.answer("compare alpha and beta fees")
check("comparison retrieves both colleges", {s["college"] for s in r["sources"]} == {"Alpha Institute (FIXTURE)", "Beta College (FIXTURE)"})
r = rag.answer("what is the fee?")
check("no college -> asks which one (no cross-college leak)", r["route"] == "need_college")
r = rag.answer("tell me about alpha placements")
check("answer cites sources even when topic data thin", r["route"] == "rag" and r["sources"])
seen = {}
r = rag.answer("alpha fee?", llm=lambda s, u: (seen.update(s=s, u=u) or "Fee is per [1]."))
check("llm path: grounding prompt + context passed", "Never invent" in seen["s"] and "11111" in seen["u"] and "Sources:" in r["answer"])
check("rank question routed to recommender, not RAG", R.route("Which colleges suit my JEE rank 50000 OBC?") == "recommend"
      and R.route("what is hostel fee at alpha") == "rag")

# content change -> update
PAGES["/a/fees"] = PAGES["/a/fees"].replace("11111", "12345")
check("changed content detected + replaced", rag.ingest_url("alpha", B + "/a/fees", "2025-26")["result"] == "updated")
r = rag.answer("alpha B.Tech CSE fee")
check("old figure gone, new figure served", "12345" in r["answer"] and "11111" not in r["answer"])

# broken URL keeps old data, marks error
del PAGES["/b/fees"]
res = rag.ingest_url("beta", B + "/b/fees", "2025-26")
check("404 handled, flagged", res["result"] == "error")
check("old beta chunks retained but source flagged failing", "http" in rag.status()["failing_sources"][0]
      and "33333" in rag.answer("beta fee")["answer"])
print("refresh_all:", [x["result"] for x in rag.refresh_all()])
print("status:", rag.status())
print(f"\n{ok} checks passed")
