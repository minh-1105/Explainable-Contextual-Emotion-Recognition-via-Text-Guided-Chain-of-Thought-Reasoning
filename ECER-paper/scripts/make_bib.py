"""Generate refs.bib from the verified metadata in logs/ref_sources*.json.

Title and authors are copied from the fetched records (arXiv API / Crossref);
an entry's 'au' list replaces them where the version of record names an author
differently (source noted beside it).  All authors are kept.  Venue strings are
fixed here by hand, pages/volume/DOI come from Crossref when the entry has one.
Only keys that main.tex actually cites are emitted.

Usage (cwd = ECER-paper): python scripts/make_bib.py
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = json.loads((ROOT / "logs" / "ref_sources.json").read_text(encoding="utf-8"))
EXTRA = json.loads((ROOT / "logs" / "ref_sources_extra.json").read_text(encoding="utf-8"))

# BibTeX-safe accents for the few non-ASCII letters found in author names.
ACCENTS = {
    "á": r"{\'a}", "é": r"{\'e}", "í": r"{\'i}", "ó": r"{\'o}", "ú": r"{\'u}",
    "ö": r'{\"o}', "ü": r'{\"u}', "ä": r'{\"a}', "ñ": r"{\~n}", "ç": r"{\c c}",
    "Á": r"{\'A}", "É": r"{\'E}", "Ö": r'{\"O}', "Ü": r'{\"U}', "ł": r"{\l}",
    "ø": r"{\o}", "š": r"{\v s}", "č": r"{\v c}", "ć": r"{\'c}", "ğ": r"{\u g}",
    "ı": r"{\i}", "ß": r"{\ss}", "è": r"{\`e}", "à": r"{\`a}", "ô": r"{\^o}",
    "ã": r"{\~a}", "ž": r"{\v z}", "ř": r"{\v r}", "ě": r"{\v e}", "ő": r'{\H o}',
    "ė": r"{\.e}", "ū": r"{\=u}",
}
PUNCT = {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2013": "--",
         "\u2014": "---", "\u00a0": " "}


def tex(s):
    for k, v in PUNCT.items():
        s = s.replace(k, v)
    out = []
    for ch in s:
        if ord(ch) < 128:
            out.append(ch)
        elif ch in ACCENTS:
            out.append(ACCENTS[ch])
        else:
            base = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode()
            if not base:
                sys.exit(f"unmapped character {ch!r} in {s!r}")
            out.append(base)
            print(f"WARN: ascii-folded {ch!r} in {s!r}", file=sys.stderr)
    s = "".join(out)
    s = s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
    return re.sub(r"\s+", " ", s).strip()


def authors(lst):
    return " and ".join(tex(a) for a in lst)


def rec(src):
    kind, ident = src
    if kind == "ax":
        r = SRC["arxiv"][ident]
        return r["title"], r["authors"], r
    if kind == "cr":
        r = SRC["crossref"][ident]
        return r["title"], r["authors"], r
    r = EXTRA[ident]
    return r["title"], r["authors"], r


def pages(p):
    return p.replace("-", "--") if p else None


# key: (entry type, title/author source, fixed fields)
# Crossref-backed entries read pages (and doi) from the record via 'pg'/'doi'.
E = {
    # --- datasets and context-aware recognition ---
    # au: PubMed 31095475 (IEEE's Crossref deposit drops the middle initial)
    "kosti2020emotic": ("article", ("cr", "10.1109/TPAMI.2019.2916866"), dict(
        au=["Ronak Kosti", "Jose M. Alvarez", "Adria Recasens", "Agata Lapedriza"],
        journal="IEEE Trans. Pattern Anal. Mach. Intell.", volume="42", number="11",
        pages="2755--2766", year="2020", doi="10.1109/TPAMI.2019.2916866")),
    "kosti2017emotic": ("inproceedings", ("cr", "10.1109/CVPRW.2017.285"), dict(
        booktitle="Proc. IEEE Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)",
        year="2017", pg=True, doi="10.1109/CVPRW.2017.285")),
    "lee2019caer": ("inproceedings", ("x", "lee2019caer"), dict(
        booktitle="Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)", year="2019", pg=True,
        doi="10.1109/iccv.2019.01024")),
    "mittal2020emoticon": ("inproceedings", ("x", "mittal2020emoticon"), dict(
        booktitle="Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)", year="2020",
        pg=True, doi="10.1109/cvpr42600.2020.01424")),
    "zhang2023emotionclip": ("inproceedings", ("x", "zhang2023emotionclip"), dict(
        booktitle="Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)", year="2023",
        pg=True, doi="10.1109/cvpr52729.2023.01821")),
    "yang2022heco": ("inproceedings", ("cr", "10.1007/978-3-031-19836-6_9"), dict(
        booktitle="Proc. Eur. Conf. Comput. Vis. (ECCV), Lecture Notes in Computer Science, "
                  "vol. 13697",
        year="2022", pg=True, doi="10.1007/978-3-031-19836-6_9")),
    "yang2024debias": ("inproceedings", ("cr", "10.1109/CVPR52733.2024.01183"), dict(
        booktitle="Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)", year="2024",
        pg=True, doi="10.1109/CVPR52733.2024.01183")),
    "li2024tscnet": ("inproceedings", ("cr", "10.1145/3664647.3680623"), dict(
        booktitle="Proc. 32nd ACM Int. Conf. Multimedia (ACM MM)", year="2024", pg=True,
        doi="10.1145/3664647.3680623")),
    # au: Springer article page (Crossref drops the "Jr." suffix)
    "luo2020arbee": ("article", ("cr", "10.1007/s11263-019-01215-y"), dict(
        au=["Yu Luo", "Jianbo Ye", "Adams, Jr., Reginald B.", "Jia Li", "Michelle G. Newman",
            "James Z. Wang"],
        journal="Int. J. Comput. Vis.", volume="128", number="1", pg=True, year="2020",
        doi="10.1007/s11263-019-01215-y")),
    "he2016resnet": ("inproceedings", ("x", "he2016resnet"), dict(
        booktitle="Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)", year="2016",
        pg=True, doi="10.1109/cvpr.2016.90")),
    # --- VLM / LLM approaches to contextual emotion ---
    "etesam2023": ("inproceedings", ("ax", "2310.19995"), dict(
        booktitle="Proc. 12th Int. Conf. Affective Comput. Intell. Interaction (ACII)",
        year="2024", pages="10--18", doi="10.1109/ACII63134.2024.00006")),
    "etesam2024": ("inproceedings", ("cr", "10.1109/IROS58592.2024.10802538"), dict(
        booktitle="Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)", year="2024",
        pg=True, doi="10.1109/IROS58592.2024.10802538")),
    "yang2023captions": ("inproceedings", ("x", "yang2023captions"), dict(
        booktitle="Proc. Int. Conf. Affective Comput. Intell. Interaction (ACII)", year="2023",
        pg=True, doi="10.1109/acii59096.2023.10388198")),
    "xenos2025vllm": ("inproceedings", ("cr", "10.1109/IJCNN64981.2025.11227260"), dict(
        booktitle="Proc. Int. Joint Conf. Neural Netw. (IJCNN)", year="2025", pg=True,
        doi="10.1109/IJCNN64981.2025.11227260")),
    "lei2025lvlm": ("inproceedings", ("ax", "2407.11300"), dict(
        booktitle="Proc. 16th Asian Conf. Mach. Learn. (ACML), PMLR vol. 260", year="2025",
        pages="111--126")),
    "zhang2024sov": ("inproceedings", ("cr", "10.18653/v1/2024.emnlp-main.257"), dict(
        booktitle="Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)", year="2024",
        pg=True, doi="10.18653/v1/2024.emnlp-main.257")),
    "bhattacharyya2025eve": ("inproceedings", ("cr", "10.18653/v1/2025.findings-naacl.97"), dict(
        booktitle="Findings Assoc. Comput. Linguistics: NAACL", year="2025", pg=True,
        doi="10.18653/v1/2025.findings-naacl.97")),
    "chen2026emocot": ("inproceedings", ("cr", "10.1007/978-981-92-3504-9_49"), dict(
        booktitle="Advanced Intelligent Computing Technology and Applications (ICIC 2026), "
                  "Lecture Notes in Computer Science, vol. 16659",
        publisher="Springer, Singapore", year="2027", pg=True,
        doi="10.1007/978-981-92-3504-9_49")),
    "saim2025elena": ("inproceedings", ("ax", "2509.19595"), dict(
        booktitle="Findings Assoc. Comput. Linguistics: EMNLP", year="2025",
        pages="23480--23495", doi="10.18653/v1/2025.findings-emnlp.1276")),
    "wu2025customizing": ("inproceedings", ("ax", "2509.21950"), dict(
        booktitle="Proc. Int. Conf. Learn. Represent. (ICLR)", year="2026",
        pages="96462--96482")),
    "she2026aica": ("inproceedings", ("ax", "2604.05900"), dict(
        booktitle="Findings Assoc. Comput. Linguistics: ACL", year="2026",
        pages="13501--13528", doi="10.18653/v1/2026.findings-acl.661")),
    "lin2025eibench": ("inproceedings", ("x", "lin2025eibench"), dict(
        booktitle="Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)",
        year="2025", pg=True, doi="10.1109/cvprw67362.2025.00515")),
    # au: NeurIPS proceedings page (arXiv omits "G.")
    "cheng2024emotionllama": ("inproceedings", ("ax", "2406.11161"), dict(
        au=["Zebang Cheng", "Zhi-Qi Cheng", "Jun-Yan He", "Jingdong Sun", "Kai Wang",
            "Yuxiang Lin", "Zheng Lian", "Xiaojiang Peng", "Alexander G. Hauptmann"],
        booktitle="Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)", volume="37", year="2024",
        pages="110805--110853", doi="10.52202/079017-3518")),
    "lian2023emer": ("misc", ("ax", "2306.15401"), dict(
        howpublished="arXiv preprint arXiv:2306.15401", year="2023")),
    "zhao2025r1omni": ("misc", ("ax", "2503.05379"), dict(
        howpublished="arXiv preprint arXiv:2503.05379", year="2025")),
    # NeurIPS 2026 poster 149766; proceedings not yet published
    "lian2026affectgptrl": ("misc", ("ax", "2605.06126"), dict(
        howpublished="accepted to Proc. Adv. Neural Inf. Process. Syst. (NeurIPS) 2026; "
                     "arXiv:2605.06126", year="2026")),
    "rha2026erv": ("inproceedings", ("x", "rha2026erv"), dict(
        booktitle="Proc. AAAI Conf. Artif. Intell.", volume="40", number="3", year="2026", pg=True,
        doi="10.1609/aaai.v40i3.37184")),
    "yang2024emollm": ("misc", ("ax", "2406.16442"), dict(
        howpublished="arXiv preprint arXiv:2406.16442", year="2024")),
    "wang2026affectomni": ("article", ("ax", "2608.26193"), dict(
        journal="IEEE Trans. Affect. Comput.", note="early access", year="2026",
        doi="10.1109/TAFFC.2026.3707634")),
    # --- reasoning, faithfulness, interpretability ---
    # au: NeurIPS proceedings page (arXiv has "Quoc Le")
    "wei2022cot": ("inproceedings", ("ax", "2201.11903"), dict(
        au=["Jason Wei", "Xuezhi Wang", "Dale Schuurmans", "Maarten Bosma", "Brian Ichter",
            "Fei Xia", "Ed Chi", "Quoc V. Le", "Denny Zhou"],
        booktitle="Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)", volume="35", year="2022",
        pages="24824--24837", doi="10.52202/068431-1800")),
    # au: NeurIPS proceedings page (arXiv has "Samuel R. Bowman")
    "turpin2023": ("inproceedings", ("ax", "2305.04388"), dict(
        au=["Miles Turpin", "Julian Michael", "Ethan Perez", "Samuel Bowman"],
        booktitle="Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)", volume="36", year="2023",
        pages="74952--74965", doi="10.52202/075280-3275")),
    "lanham2023": ("misc", ("ax", "2307.13702"), dict(
        howpublished="arXiv preprint arXiv:2307.13702", year="2023")),
    "deyoung2020eraser": ("inproceedings", ("cr", "10.18653/v1/2020.acl-main.408"), dict(
        booktitle="Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL)", year="2020",
        pg=True, doi="10.18653/v1/2020.acl-main.408")),
    "jacovi2020": ("inproceedings", ("cr", "10.18653/v1/2020.acl-main.386"), dict(
        booktitle="Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL)", year="2020",
        pg=True, doi="10.18653/v1/2020.acl-main.386")),
    "parcalabescu2024": ("inproceedings", ("ax", "2404.18624"), dict(
        booktitle="Proc. Int. Conf. Learn. Represent. (ICLR)", year="2025",
        pages="21634--21663")),
    "atanasova2023": ("inproceedings", ("ax", "2305.18029"), dict(
        booktitle="Proc. 61st Annu. Meeting Assoc. Comput. Linguistics (Vol. 2: Short Papers)",
        year="2023", pages="283--294", doi="10.18653/v1/2023.acl-short.25")),
    "hase2020las": ("inproceedings", ("x", "hase2020las"), dict(
        booktitle="Findings Assoc. Comput. Linguistics: EMNLP", year="2020", pg=True,
        doi="10.18653/v1/2020.findings-emnlp.390")),
    "chen2023cs": ("inproceedings", ("ax", "2307.08678"), dict(
        booktitle="Proc. 41st Int. Conf. Mach. Learn. (ICML), PMLR vol. 235", year="2024",
        pages="7880--7904")),
    "doshivelez2017": ("misc", ("ax", "1702.08608"), dict(
        howpublished="arXiv preprint arXiv:1702.08608", year="2017")),
    "huynh2026facr": ("misc", ("ax", "2606.15779"), dict(
        howpublished="arXiv preprint arXiv:2606.15779", year="2026")),
    "ding2026edct": ("misc", ("ax", "2609.17953"), dict(
        howpublished="arXiv preprint arXiv:2609.17953", year="2026")),
    "menzat2026vct": ("misc", ("ax", "2609.06704"), dict(
        howpublished="accepted to Findings Assoc. Comput. Linguistics: EMNLP 2026; "
                     "arXiv:2609.06704", year="2026")),
    "han2026oppo": ("inproceedings", ("ax", "2606.25325"), dict(
        booktitle="Proc. 43rd Int. Conf. Mach. Learn. (ICML), PMLR vol. 306", year="2026",
        pages="40269--40294")),
    "ye2026dspo": ("misc", ("ax", "2609.36775"), dict(
        howpublished="arXiv preprint arXiv:2609.36775", year="2026")),
    # --- annotation and psychology ---
    "martinez2023reliability": ("article", ("cr", "10.1038/s41597-023-02435-1"), dict(
        journal="Sci. Data", volume="10", number="1", pages="538", year="2023",
        doi="10.1038/s41597-023-02435-1")),
    "barrett2019": ("article", ("cr", "10.1177/1529100619832930"), dict(
        journal="Psychol. Sci. Public Interest", volume="20", number="1", pages="1--68",
        year="2019", doi="10.1177/1529100619832930")),
    "barrett2011": ("article", ("cr", "10.1177/0963721411422522"), dict(
        journal="Curr. Dir. Psychol. Sci.", volume="20", number="5", pages="286--290",
        year="2011", doi="10.1177/0963721411422522")),
    # --- scoring / readout, grounding, models ---
    "ma2025mlc": ("inproceedings", ("cr", "10.18653/v1/2025.emnlp-main.126"), dict(
        booktitle="Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)", year="2025",
        pg=True, doi="10.18653/v1/2025.emnlp-main.126")),
    "hallmen2026readout": ("misc", ("ax", "2610.08162"), dict(
        howpublished="arXiv preprint arXiv:2610.08162", year="2026")),
    "kim2023vcr": ("inproceedings", ("cr", "10.18653/v1/2023.ijcnlp-main.66"), dict(
        booktitle="Proc. Int. Joint Conf. Natural Lang. Process. and Asia-Pacific Chapter "
                  "Assoc. Comput. Linguistics (IJCNLP-AACL) (Vol. 1: Long Papers)", year="2023",
        pg=True,
        doi="10.18653/v1/2023.ijcnlp-main.66")),
    "bai2025qwen25vl": ("misc", ("ax", "2502.13923"), dict(
        howpublished="arXiv preprint arXiv:2502.13923", year="2025")),
    "radford2021clip": ("misc", ("ax", "2103.00020"), dict(
        howpublished="arXiv preprint arXiv:2103.00020", year="2021")),
    "liu2023llava": ("misc", ("ax", "2304.08485"), dict(
        howpublished="arXiv preprint arXiv:2304.08485", year="2023")),
}


def cited_keys(tex_path):
    txt = re.sub(r"(?m)(?<!\\)%.*$", "", tex_path.read_text(encoding="utf-8"))
    keys = []
    for m in re.finditer(r"\\cite[a-zA-Z]*\{([^}]*)\}", txt):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)
    return keys


def main():
    keys = cited_keys(ROOT / "main.tex")
    unknown = [k for k in keys if k not in E]
    if unknown:
        sys.exit(f"cited but undefined: {unknown}")
    out = ["% Generated by scripts/make_bib.py from logs/ref_sources*.json. Do not hand-edit.\n"]
    for k in keys:
        typ, src, f = E[k]
        title, auth, r = rec(src)
        f = dict(f)
        if f.pop("pg", False):
            f["pages"] = pages(r["pages"])
        if r.get("pages") is None and "pages" not in f and src[0] in ("cr", "x"):
            pass
        auth = f.pop("au", auth)
        fields = [("author", authors(auth)), ("title", "{" + tex(title) + "}")]
        order = ["journal", "booktitle", "howpublished", "publisher", "volume", "number", "pages", "year", "note", "doi"]
        for o in order:
            if o in f:
                v = f[o]
                fields.append((o, v if o == "year" else tex(v) if o != "doi" else v))
        out.append(f"@{typ}{{{k},")
        for name, val in fields:
            if name == "title":
                out.append(f"  title = {{{val}}},")
            else:
                out.append(f"  {name} = {{{val}}},")
        out[-1] = out[-1].rstrip(",")
        out.append("}\n")
    (ROOT / "refs.bib").write_text("\n".join(out), encoding="utf-8")
    print(f"wrote refs.bib with {len(keys)} entries")


if __name__ == "__main__":
    main()
