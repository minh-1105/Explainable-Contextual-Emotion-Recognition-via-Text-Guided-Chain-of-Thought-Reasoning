"""Fetch bibliographic metadata for every candidate reference from public sources.

Sources: arXiv API (arXiv IDs), Crossref (DOIs), ACL Anthology BibTeX (ACL IDs),
PMLR page (Lei et al.). Output: logs/ref_sources.json. Nothing here is written by hand;
verify_refs.py later compares refs.bib against this file.
"""
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "logs" / "ref_sources.json"

ARXIV_IDS = """
2003.13401 1908.05913 2003.06692 2404.17205 2403.05963 2305.13500 1808.09568
2310.19995 2405.08992 2309.13136 2407.11300 2404.07078 2410.02244 2509.19595
2406.16442 2502.05660 2509.21950 2306.15401 2406.11161 2503.05379 2605.06126
2510.23506 2504.07521 2604.05900 2307.13702 2305.04388 2404.18624 2606.15779
2609.17953 2609.06704 2609.36775 2606.25325 2307.08678 2305.18029 2010.04119
2610.08162 2505.17510 2201.11903 2502.13923 1512.03385 2103.00020 2304.08485
2608.26193 1702.08608
""".split()

DOIS = [
    "10.1109/TPAMI.2019.2916866",       # Kosti 2020
    "10.18653/v1/2020.acl-main.408",   # ERASER
    "10.1109/CVPRW.2017.285",           # Kosti 2017
    "10.1007/978-3-031-19836-6_9",      # HECO
    "10.1038/s41597-023-02435-1",       # Martinez-Miwa
    "10.1177/1529100619832930",         # Barrett 2019
    "10.1177/0963721411422522",         # Barrett 2011
    "10.1007/978-981-92-3504-9_49",     # EmoCoT
    "10.1109/IROS58592.2024.10802538",  # Etesam IROS
    "10.1145/3664647.3680623",          # TSCNet
    "10.1109/CVPR52733.2024.01183",     # debias
    "10.18653/v1/2024.emnlp-main.257",  # SoV
    "10.18653/v1/2023.ijcnlp-main.66",  # Kim VCR-CS
    "10.18653/v1/2020.acl-main.386",    # Jacovi
    "10.18653/v1/2025.emnlp-main.126",  # Ma
    "10.18653/v1/2025.findings-naacl.97",  # Bhattacharyya
    "10.1109/IJCNN64981.2025.11227260", # Xenos
    "10.1007/s11263-019-01215-y",       # ARBEE IJCV (checked via Crossref)
]

ACL_IDS = ["2023.ijcnlp-main.66", "2020.acl-main.386", "2024.emnlp-main.257",
           "2025.emnlp-main.126", "2025.findings-naacl.97"]

NS = {"a": "http://www.w3.org/2005/Atom", "ar": "http://arxiv.org/schemas/atom"}


def fetch_arxiv(ids):
    out = {}
    for i in range(0, len(ids), 20):
        batch = ids[i:i + 20]
        r = requests.get("https://export.arxiv.org/api/query",
                         params={"id_list": ",".join(batch), "max_results": len(batch)}, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        for e in root.findall("a:entry", NS):
            aid = e.find("a:id", NS).text.split("/abs/")[-1]
            aid = re.sub(r"v\d+$", "", aid)
            out[aid] = {
                "title": " ".join(e.find("a:title", NS).text.split()),
                "authors": [a.find("a:name", NS).text for a in e.findall("a:author", NS)],
                "published": e.find("a:published", NS).text[:10],
                "updated": e.find("a:updated", NS).text[:10],
                "journal_ref": (e.find("ar:journal_ref", NS).text if e.find("ar:journal_ref", NS) is not None else None),
                "doi": (e.find("ar:doi", NS).text if e.find("ar:doi", NS) is not None else None),
            }
        time.sleep(3)
    return out


def fetch_crossref(dois):
    out = {}
    for d in dois:
        r = requests.get(f"https://api.crossref.org/works/{d}", timeout=60,
                         headers={"User-Agent": "ecer-paper-ref-check/1.0 (mailto:none@example.com)"})
        if r.status_code != 200:
            out[d] = {"error": r.status_code}
            continue
        m = r.json()["message"]
        issued = (m.get("issued", {}).get("date-parts") or [[None]])[0]
        out[d] = {
            "title": " ".join((m.get("title") or [""])[0].split()),
            "authors": [f"{a.get('given', '')} {a.get('family', a.get('name', ''))}".strip() for a in m.get("author", [])],
            "container": (m.get("container-title") or [""])[0],
            "year": issued[0],
            "volume": m.get("volume"), "issue": m.get("issue"), "pages": m.get("page"),
            "publisher": m.get("publisher"), "type": m.get("type"),
        }
        time.sleep(0.5)
    return out


def fetch_acl(ids):
    out = {}
    for i in ids:
        r = requests.get(f"https://aclanthology.org/{i}.bib", timeout=60)
        out[i] = r.text if r.status_code == 200 else f"ERROR {r.status_code}"
        time.sleep(0.5)
    return out


def fetch_pmlr():
    r = requests.get("https://proceedings.mlr.press/v260/lei25a.html", timeout=60)
    txt = r.text if r.status_code == 200 else f"ERROR {r.status_code}"
    keep = [l.strip() for l in txt.splitlines()
            if re.search(r"citation_(title|author|publication_date|firstpage|lastpage|conference_title|volume)", l)]
    return keep


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    data = {"arxiv": fetch_arxiv(ARXIV_IDS), "crossref": fetch_crossref(DOIS),
            "acl_bib": fetch_acl(ACL_IDS), "pmlr_lei": fetch_pmlr()}
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    missing = [i for i in ARXIV_IDS if i not in data["arxiv"]]
    print("arxiv fetched:", len(data["arxiv"]), "missing:", missing)
    print("crossref errors:", [d for d, v in data["crossref"].items() if "error" in v])
