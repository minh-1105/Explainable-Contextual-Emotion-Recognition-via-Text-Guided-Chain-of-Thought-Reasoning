import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
PAPERS = {
    "emotic": "2003.13401",
    "caer": "1908.05913",
    "emoticon": "2003.06692",
    "arbee": "1808.09568",
    "emotionclip": "2305.13500",
    "debias": "2403.05963",
    "decoupled": "2404.17205",
    "captions": "2309.13136",
    "etom": "2310.19995",
    "contextvlm": "2405.08992",
    "lei": "2407.11300",
    "visualprompt": "2410.02244",
    "evaluatevlm": "2502.05660",
    "commonsense": "2404.07078",
    "emotionllama": "2406.11161",
    "emollm": "2406.16442",
    "emer": "2306.15401",
    "r1omni": "2503.05379",
    "cot": "2201.11903",
    "turpin": "2305.04388",
    "lanham": "2307.13702",
    "faithfulnesstests": "2305.18029",
    "decoderfaithfulness": "2404.18624",
    "simulatability": "2010.04119",
    "counterfactual": "2307.08678",
    "interpretability": "1702.08608",
    "resnet": "1512.03385",
    "bert": "1810.04805",
    "vit": "2010.11929",
    "transformer": "1706.03762",
    "adamw": "1711.05101",
    "eraser": "1911.03429",
    "faithfulness": "2004.03685",
}


class AbstractPage(HTMLParser):
    def __init__(self):
        super().__init__()
        self.metadata = {}
        self.abstract = []
        self.in_abstract = False

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "meta" and "name" in values and "content" in values:
            self.metadata.setdefault(values["name"], []).append(values["content"])
        if tag == "blockquote" and "abstract" in values.get("class", "").split():
            self.in_abstract = True

    def handle_endtag(self, tag):
        if tag == "blockquote":
            self.in_abstract = False

    def handle_data(self, data):
        if self.in_abstract:
            self.abstract.append(data)


def fetch(identifier):
    url = f"https://arxiv.org/abs/{identifier}"
    response = requests.get(url, headers={"User-Agent": "ReferenceReader/1.0"}, timeout=30)
    response.raise_for_status()
    page = AbstractPage()
    page.feed(response.text)
    if not page.abstract:
        raise ValueError(f"No abstract: {identifier}")
    text = re.sub(r"^Abstract:\s*", "", " ".join(" ".join(page.abstract).split()))
    title = page.metadata["citation_title"][0]
    authors = page.metadata["citation_author"]
    date = page.metadata["citation_date"][0]
    if not text or not authors or identifier not in response.url:
        raise ValueError(f"Invalid reference response: {identifier}")
    return {"arxiv": identifier, "url": url, "title": title, "authors": authors,
            "date": date, "abstract": text}


def main():
    records = {}
    for key, identifier in PAPERS.items():
        records[key] = fetch(identifier)
        print(key, records[key]["title"], flush=True)
        time.sleep(0.4)
    (ROOT / "logs/references.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Fetched {len(records)} abstracts and metadata records")


if __name__ == "__main__":
    main()
