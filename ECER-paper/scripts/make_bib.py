import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def tex(text):
    accents = {"ç": r"{\c c}", "ı": r"{\i}", "Ö": r'{\"O}', "ö": r'{\"o}', "ü": r'{\"u}', "é": r"{\'e}", "ė": r"{\.e}", "ū": r"{\=u}", "š": r"{\v s}", "á": r"{\'a}", "à": r"{\`a}", "ñ": r"{\~n}"}
    text = text.replace("’", "'").replace("–", "--").replace("—", "---")
    output = []
    for char in text:
        if char in accents:
            output.append(accents[char])
        elif ord(char) < 128:
            output.append({"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}.get(char, char))
        else:
            folded = unicodedata.normalize("NFKD", char).encode("ascii", "ignore").decode()
            if not folded:
                raise ValueError(f"Unmapped bibliography character: {char!r}")
            output.append(folded)
    return " ".join("".join(output).split())


def render(source):
    records = json.loads((ROOT / "logs/references.json").read_text(encoding="utf-8"))
    keys = list(dict.fromkeys(key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", source) for key in group.split(",")))
    if set(keys) != set(records):
        raise ValueError(f"Reference mismatch: missing={set(keys)-set(records)}, unused={set(records)-set(keys)}")
    entries = []
    for key in keys:
        record = records[key]
        if not record["abstract"].strip():
            raise ValueError(f"Missing abstract: {key}")
        fields = {"author": " and ".join(tex(author) for author in record["authors"]), "title": "{" + tex(record["title"]) + "}", "howpublished": "arXiv preprint arXiv:" + record["arxiv"], "year": re.match(r"\d{4}", record["date"])[0]}
        entries.append("@misc{" + key + ",\n" + ",\n".join(f"  {name} = {{{value}}}" for name, value in fields.items()) + "\n}")
    return "\n\n".join(entries) + "\n"


if __name__ == "__main__":
    output = render((ROOT / "main.tex").read_text(encoding="utf-8"))
    (ROOT / "refs.bib").write_text(output, encoding="ascii")
    print(f"Generated {output.count('@misc{')} references from fetched arXiv metadata")
