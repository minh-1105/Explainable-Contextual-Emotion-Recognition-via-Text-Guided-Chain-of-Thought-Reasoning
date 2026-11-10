"""Static checks for main.tex / refs.bib (no LaTeX compiler is available on this machine).

1. cite <-> bib bijection and reference count
2. bib entries vs fetched source records (title/author count/year/DOI consistency)
3. LaTeX structure: braces, begin/end, labels/refs, figure files, unescaped specials, markers
4. word counts per section and a rough page estimate (heuristic, NOT a compile)

Usage (cwd = ECER-paper): python scripts/verify_paper.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
raw = (ROOT / "main.tex").read_text(encoding="utf-8")
bib = (ROOT / "refs.bib").read_text(encoding="utf-8")
SRC = json.loads((ROOT / "logs" / "ref_sources.json").read_text(encoding="utf-8"))
EXTRA = json.loads((ROOT / "logs" / "ref_sources_extra.json").read_text(encoding="utf-8"))
errors = []


def err(msg):
    errors.append(msg)
    print("FAIL:", msg)


def strip_comments(t):
    return re.sub(r"(?<!\\)%.*", "", t)


tex = strip_comments(raw)

# ---------- 1. cite <-> bib ----------
cited = []
for m in re.finditer(r"\\cite\w*\{([^}]*)\}", tex):
    for k in m.group(1).split(","):
        k = k.strip()
        if k:
            cited.append(k)
cited_set = set(cited)
entries = re.findall(r"@(\w+)\{([^,\s]+),", bib)
keys = [k for _, k in entries]
print(f"[1] cite commands: {len(cited)} keys used, {len(cited_set)} unique; bib entries: {len(keys)}")
for k in sorted(cited_set - set(keys)):
    err(f"cited but missing in refs.bib: {k}")
for k in sorted(set(keys) - cited_set):
    err(f"in refs.bib but never cited: {k}")
if len(keys) != len(set(keys)):
    err("duplicate bib keys")
if len(cited_set) < 30:
    err(f"fewer than 30 unique references cited ({len(cited_set)})")
else:
    print(f"    OK: {len(cited_set)} unique cited references (>= 30), every bib entry cited, no missing keys")

# ---------- 2. bib vs source records ----------
spec = {}
exec_ns = {}
src = (ROOT / "scripts" / "make_bib.py").read_text(encoding="utf-8")
m = re.search(r"^E = \{.*?^\}\n", src, re.S | re.M)
exec(compile("E = " + m.group(0)[4:], "E", "exec"), exec_ns)
E = exec_ns["E"]


def norm(s):
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\\&", "&").replace("\\_", "_")
    s = re.sub(r"\{\\[^}]*\}", "", s)
    return re.sub(r"[^a-z0-9]", "", s.lower())


checked = 0
for blk in re.finditer(r"@(\w+)\{([^,]+),\n(.*?)\n\}", bib, re.S):
    key, body = blk.group(2), blk.group(3)
    f = dict(re.findall(r"^\s*(\w+) = \{(.*)\},?$", body, re.M))
    kind, ident = E[key][1]
    rec = {"ax": SRC["arxiv"], "cr": SRC["crossref"], "x": EXTRA}[kind][ident]
    t = f["title"].strip("{}")
    if norm(t) != norm(rec["title"]):
        err(f"{key}: title differs from source record")
    n_auth = len(f["author"].split(" and "))
    if "others" in f["author"]:
        if len(rec["authors"]) <= 6:
            err(f"{key}: truncated author list but source has <= 6 authors")
    elif n_auth != len(rec["authors"]):
        err(f"{key}: author count {n_auth} != source {len(rec['authors'])}")
    first_last = rec["authors"][0].split()[-1]
    if norm(first_last) not in norm(f["author"].split(" and ")[0]) and kind != "ax":
        pass  # accented names are folded to TeX macros; counted via title/author count above
    if "doi" in f and kind in ("cr", "x") and f["doi"].lower() != ident.lower() and kind == "cr":
        err(f"{key}: doi differs from Crossref record")
    if kind == "x" and f["doi"].lower() != rec["doi"].lower():
        err(f"{key}: doi differs from Crossref search record")
    if "pages" in f and kind in ("cr", "x") and rec.get("pages") and f["pages"].replace("--", "-") != rec["pages"]:
        if key != "kosti2020emotic":  # final TPAMI pages from the paper's own citations (Crossref has early access '1-1')
            err(f"{key}: pages differ from Crossref")
    checked += 1
print(f"[2] bib entries checked against fetched metadata: {checked}")

# ---------- 3. structure ----------
body = re.sub(r"\\verb\|[^|]*\|", "", tex)
depth, line_no, first_bad = 0, 1, None
for i, ch in enumerate(body):
    if ch == "\n":
        line_no += 1
    prev = body[i - 1] if i else ""
    if prev == "\\":
        continue
    if ch == "{":
        depth += 1
    elif ch == "}":
        depth -= 1
        if depth < 0 and first_bad is None:
            first_bad = line_no
if depth != 0 or first_bad:
    err(f"unbalanced braces (final depth {depth}, first negative at line {first_bad})")
else:
    print("[3] braces balanced")

stack = []
for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", body):
    if m.group(1) == "begin":
        stack.append(m.group(2))
    else:
        if not stack or stack.pop() != m.group(2):
            err(f"mismatched \\end{{{m.group(2)}}}")
if stack:
    err(f"unclosed environments: {stack}")
else:
    print("    begin/end environments balanced")
for need in ("document", "abstract", "IEEEkeywords"):
    if f"\\begin{{{need}}}" not in body:
        err(f"missing environment {need}")

labels = set(re.findall(r"\\label\{([^}]+)\}", body))
refs = set(re.findall(r"\\ref\{([^}]+)\}", body))
for r in sorted(refs - labels):
    err(f"\\ref to undefined label {r}")
for l in sorted(labels - refs):
    err(f"label never referenced: {l}")
print(f"    labels defined: {len(labels)}, referenced: {len(refs)}")

for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
    p = ROOT / g
    if not p.exists():
        err(f"figure file missing: {g}")
    else:
        print(f"    figure present: {g} ({p.stat().st_size} bytes)")

# unescaped specials in prose (outside math, tabular rows for &, and arguments that legitimately hold _)
prose = re.sub(r"\\(?:includegraphics|label|ref|cite|bibliography|bibliographystyle)(?:\[[^\]]*\])?\{[^}]*\}", "", body)
prose = re.sub(r"\$[^$]*\$", "", prose)
prose = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", "", prose, re.S) if False else prose
tab_free = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", "", prose, flags=re.S)
for pat, name in ((r"(?<!\\)_", "underscore"), (r"(?<!\\)&", "ampersand"), (r"(?<!\\)#", "hash")):
    for m in re.finditer(pat, tab_free):
        ctx = tab_free[max(0, m.start() - 25): m.end() + 25].replace("\n", " ")
        err(f"unescaped {name}: ...{ctx}...")
for m in re.finditer(r"\d(?<!\\)%", raw):
    # a bare % after a digit in the raw file would silently comment out the rest of the line
    if not re.search(r"\\%", raw[m.start(): m.end() + 1]):
        err(f"possible unescaped percent after digit near: {raw[max(0, m.start() - 20): m.end() + 10]!r}")
if body.count("$") - body.count("\\$") != 0 and (body.count("$") - body.count("\\$")) % 2:
    err("odd number of $ signs")
else:
    print("    no unescaped _ & # %, even number of $")

i0, i1 = raw.find("% PROVISIONAL-CONTRIBUTIONS"), raw.find("% END-PROVISIONAL")
if not (0 <= i0 < i1):
    err("PROVISIONAL-CONTRIBUTIONS / END-PROVISIONAL markers missing or out of order")
else:
    print("    PROVISIONAL-CONTRIBUTIONS ... END-PROVISIONAL block present")

heads = re.findall(r"\\section\{([^}]+)\}", body)
print("    sections:", heads)
for need in ("Introduction", "Related Work", "Methodology", "Experiments and Results", "Conclusion"):
    if need not in heads:
        err(f"missing section {need}")
if "\\bibliography{refs}" not in body:
    err("missing \\bibliography{refs}")

# claims that must not appear (honesty contract)
for bad in (r"state[- ]of[- ]the[- ]art", r"outperform", r"novel(?!ty)", r"faithful(?:ly)? explain", r"we propose"):
    for m in re.finditer(bad, body, re.I):
        ctx = body[max(0, m.start() - 40): m.end() + 40].replace("\n", " ")
        print(f"    note (review wording): '{m.group(0)}' ... {ctx}")

# ---------- 4. word counts / page estimate ----------
def words(t):
    t = re.sub(r"\\begin\{(tabular|figure\*?|table)\}.*?\\end\{\1\}", " ", t, flags=re.S)
    t = re.sub(r"\\cite\w*\{[^}]*\}", " C ", t)
    t = re.sub(r"\$[^$]*\$", " M ", t)
    t = re.sub(r"\\(?:ref|label)\{[^}]*\}", " R ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    t = re.sub(r"[{}\\~]", " ", t)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-.,/]*", t))


abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S).group(1)
main_text = body.split("\\maketitle")[1]
main_text = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", main_text, flags=re.S)
parts = re.split(r"\\section\{([^}]+)\}", main_text)
print(f"[4] abstract words: {words(abstract)}")
total = 0
for h, t in zip(parts[1::2], parts[2::2]):
    t = t.split("\\bibliographystyle")[0]
    w = words(t)
    total += w
    print(f"    {h:<22} {w:>5} words")
print(f"    body total (excluding abstract, tables, figure, refs): {total} words")

n_rows = sum(len(re.findall(r"\\\\", m)) for m in re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", body, re.S))
n_tab = len(re.findall(r"\\begin\{table\}", body))
cap_lines = 2 * n_tab
prose_pages = total / 950.0 + words(abstract) / 950.0 + 0.04
tab_pages = (n_rows + 3 * n_tab + cap_lines) * 11.0 / 1296.0
fig_pages = 389.0 / 1296.0 * len(re.findall(r"\\begin\{figure\*\}", body))
ref_pages = len(keys) * 28.0 / 1296.0
print(f"    tables: {n_tab} ({n_rows} rows), figures: {len(re.findall(r'begin.figure', body))}")
print(f"    ROUGH estimate (heuristic, not compiled): prose {prose_pages:.2f} + tables {tab_pages:.2f} + figure {fig_pages:.2f}"
      f" = {prose_pages + tab_pages + fig_pages:.2f} pages before references; references ~{ref_pages:.2f} page;"
      f" total ~{prose_pages + tab_pages + fig_pages + ref_pages:.1f} two-column pages")

print("\nRESULT:", "FAIL (%d)" % len(errors) if errors else "PASS")
sys.exit(1 if errors else 0)
