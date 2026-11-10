"""Print a markdown table: each cited reference -> identifier and how it was verified.
Usage (cwd = ECER-paper): python scripts/ref_table.py >> provenance.md
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = json.loads((ROOT / "logs" / "ref_sources.json").read_text(encoding="utf-8"))
ns = {}
code = (ROOT / "scripts" / "make_bib.py").read_text(encoding="utf-8")
exec(compile("E = " + re.search(r"^E = \{.*?^\}\n", code, re.S | re.M).group(0)[4:], "E", "exec"), ns)
E = ns["E"]
tex = re.sub(r"(?<!\\)%.*", "", (ROOT / "main.tex").read_text(encoding="utf-8"))
order = []
for m in re.finditer(r"\\cite\w*\{([^}]*)\}", tex):
    for k in m.group(1).split(","):
        if k.strip() not in order:
            order.append(k.strip())

print("| # | Key | Identifier | Verified through |")
print("|---|-----|------------|------------------|")
for i, k in enumerate(order, 1):
    typ, (kind, ident), f = E[k]
    via = []
    if kind == "ax":
        ident_s = f"arXiv:{ident}"
        if typ == "misc":
            via.append("arXiv API record (title, authors, date)")
        else:
            via.append("arXiv API record (title, authors); venue, pages, DOI from the published "
                       "version (section 4)")
    elif kind == "cr":
        ident_s = f"DOI {ident}"
        via.append("Crossref record (title, authors, venue, pages)")
        if ident.startswith("10.18653/v1/"):
            acl = ident.split("/v1/")[1]
            if acl in SRC["acl_bib"]:
                via.append("ACL Anthology .bib")
    else:
        ident_s = f"DOI {f['doi']}"
        via.append("Crossref title search (exact normalized title + year)")
    if k == "lei2025lvlm":
        via.append("PMLR page metadata (v260, pp. 111-126, 2025/01/14)")
    if k == "chen2026emocot":
        via.append("Springer page (abstract and reference list only; full text paywalled)")
    if k == "etesam2024":
        via.append("full PDF read")
    if k == "lei2025lvlm":
        via.append("full PDF read (CoT statements)")
    if "au" in f:
        via.append("author names from the version of record (section 4)")
    if typ == "misc" and f["howpublished"].startswith("accepted"):
        via.append("accepted, proceedings not yet published; cited as accepted (section 4)")
    elif typ == "misc":
        via.append("cited as arXiv preprint; no published version found (section 4)")
    print(f"| {i} | `{k}` | {ident_s} | {'; '.join(via)} |")
print(f"\nTotal cited references: {len(order)}")
