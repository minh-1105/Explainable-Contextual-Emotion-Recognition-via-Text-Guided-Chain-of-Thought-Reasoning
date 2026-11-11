import argparse
import contextlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

sys.dont_write_bytecode = True
from compile_paper import compile_paper
from evidence import ROOT, REPO, RUN, collect
from make_bib import render
from make_figures import build as build_figures

PATTERN = r"\\ev\{([^{}]+)\}\{([^{}]*)\}"


HELPERS = (
    r"\newcommand{\ev}[2]{#2}",
    r"\newcommand{\mf}{macro-F\textsubscript{1}}",
    r"\newcommand{\mi}{micro-F\textsubscript{1}}",
    r"\newcommand{\tablegap}{\setlength{\tabcolsep}{4pt}}",
)
COMMANDS = {
    "%", "_", "\\", "IEEEauthorblockA", "IEEEauthorblockN", "arg", "author", "begin",
    "bottomrule", "caption", "centering", "cite", "emph", "end", "ev", "frac", "in",
    "includegraphics", "label", "maketitle", "max", "mf", "mi", "midrule", "pm", "ref",
    "section", "small", "subsection", "sum", "tablegap", "texttt", "textwidth", "title", "toprule",
}
ENVIRONMENTS = {"document", "abstract", "IEEEkeywords", "equation", "table", "table*", "figure*", "tabular"}


def group_end(source, opening):
    depth = 1
    for token in re.finditer(r"\\.|[{}]", source[opening + 1:]):
        if token[0] == "{":
            depth += 1
        elif token[0] == "}":
            depth -= 1
            if depth == 0:
                return opening + token.end() + 1
    raise ValueError("Unclosed source group")


def validate_tex(source):
    if "^^" in source or re.search(r"(?<!\\)%", source):
        raise ValueError("TeX character escapes and comments are outside the paper source grammar")
    unbound = source
    for helper in HELPERS:
        if unbound.count(helper) != 1:
            raise ValueError("Structural helper must have exactly one fixed definition: " + helper)
        unbound = unbound.replace(helper, "", 1)
    unbound = re.sub(r"\\documentclass\[conference\]\{IEEEtran\}", "", unbound)
    unbound = re.sub(r"\\usepackage\{(?:cite|amsmath,amssymb|graphicx|booktabs|textcomp)\}", "", unbound)
    for package in (r"\usepackage[utf8]{inputenc}", r"\usepackage[T5]{fontenc}", r"\usepackage[vietnamese]{babel}"):
        unbound = unbound.replace(package, "")
    unbound = unbound.replace(r"\renewcommand{\IEEEkeywordsname}{Từ khóa}", "", 1)
    unbound = unbound.replace(r"\bibliographystyle{IEEEtran}", "").replace(r"\bibliography{refs}", "")
    for token in re.finditer(r"\\([A-Za-z]+|[^\n])", unbound):
        if token[1] not in COMMANDS:
            raise ValueError("unsupported TeX command: " + token[0])
    environments = re.findall(r"\\(?:begin|end)\s*\{([A-Za-z*]+)\}", unbound)
    if len(environments) != len(re.findall(r"\\(?:begin|end)\b", unbound)):
        raise ValueError("unsupported TeX environment invocation")
    for name in environments:
        if name not in ENVIRONMENTS:
            raise ValueError("unsupported TeX environment: " + name)
    author = re.search(r"\\author\{", unbound)
    if author:
        end = group_end(unbound, author.end() - 1)
        block = unbound[author.end():end]
        block = re.sub(r"[A-Za-z0-9._+-]+@[A-Za-z0-9.-]+\.[A-Za-z]+", "", block)
        unbound = unbound[:author.end()] + block + unbound[end:]
    unbound = re.sub(PATTERN, "", unbound)
    if re.search(r"\\ev\b", unbound):
        raise ValueError("malformed evidence binding")
    unbound = re.sub(r"\\(?:cite|ref|label)\{[A-Za-z0-9:_,.-]+\}", "", unbound)
    unbound = re.sub(r"\\includegraphics\[width=\\textwidth\]\{figures/[A-Za-z0-9_-]+\.png\}", "", unbound)
    literal = re.search(r".{0,30}\d.{0,30}", unbound)
    if literal:
        raise ValueError("unbound numeric literal: " + literal[0])


def check_alignment(sources):
    signatures = []
    for source in sources:
        body = source.split(r"\maketitle", 1)[1]
        structure = re.findall(r"\\(?:section|subsection|caption)\{|\\(?:begin|end)\{[^{}]+\}|\\(?:label|ref|cite)\{[^{}]+\}|\\includegraphics\[[^]]+\]\{[^{}]+\}|\\ev\{[^{}]+\}\{[^{}]*\}|\\\\|&", body)
        equations = [re.sub(r"\s+", "", equation) for pair in re.findall(r"\\begin\{equation\}(.*?)\\end\{equation\}|\$([^$]+)\$", body, re.S) for equation in pair if equation]
        metrics = tuple(body.count(macro) for macro in (r"\mf", r"\mi"))
        signatures.append((structure, equations, metrics))
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("paper alignment differs: structure, bindings, citations, figures, or equations")
    print("PASS: papers align in structure, ordered bindings/citations/figures, and equations")


def check_figures(paths, sources, evidence_root):
    assets = set()
    for path, source in zip(paths, sources):
        references = re.findall(r"\\includegraphics\[width=\\textwidth\]\{(figures/[A-Za-z0-9_-]+\.png)\}", source)
        if len(references) != len(re.findall(r"\\includegraphics\b", source)):
            raise ValueError("unsupported figure inclusion")
        assets.update((path.parent / relative, relative) for relative in references)
    with tempfile.TemporaryDirectory(prefix="ecer-figure-check-") as directory:
        generated = Path(directory)
        build_figures(figure_dir=generated, repo=evidence_root)
        for actual, relative in sorted(assets):
            expected = generated / Path(relative).name
            if not expected.is_file() or actual.read_bytes() != expected.read_bytes():
                raise ValueError("figure differs from generator: " + relative)
    print(f"PASS: {len(assets)} shared figures match fresh generator output from current sources")


def check_paper(path, source, values):
    occurrences = list(re.finditer(PATTERN, source))
    errors = []
    validate_tex(source)
    if not occurrences:
        errors.append("no evidence bindings")
    for match in occurrences:
        name, actual = match.groups()
        expected = values.get(name)
        if expected is None:
            errors.append(f"unknown evidence key: {name}")
        elif actual != expected["text"]:
            errors.append(f"{name}: paper={actual}, source={expected['text']} ({expected['source']}:{expected['key']})")
    citations = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", source) for key in group.split(",")}
    if len(citations) < 30:
        errors.append(f"fewer than 30 cited references: {len(citations)}")
    if errors:
        raise ValueError(path.name + ": " + "\n".join(errors))
    if (path.parent / "refs.bib").read_text(encoding="ascii") != render(source):
        raise ValueError(path.name + ": refs.bib differs from generated metadata")
    print(f"PASS {path.name}: {len(occurrences)} evidence bindings match source values; no unbound numeric literals")
    print(f"PASS {path.name}: {len(citations)} cited references; refs.bib matches generator from logs/references.json")


def check(paths, evidence_root):
    sources = [path.read_text(encoding="utf-8-sig") for path in paths]
    values = collect(evidence_root)
    for path, source in zip(paths, sources):
        check_paper(path, source, values)
    check_alignment(sources)
    check_figures(paths, sources, evidence_root)
    for path in paths:
        compile_paper(source=path)


def prove(paths, evidence_root):
    with tempfile.TemporaryDirectory(prefix="ecer-number-proof-") as temporary:
        scratch = Path(temporary)
        candidates = [scratch / path.name for path in paths]
        sources = [path.read_text(encoding="utf-8-sig") for path in paths]
        for candidate, source in zip(candidates, sources):
            candidate.write_text(source, encoding="utf-8")
        shutil.copyfile(paths[0].parent / "refs.bib", scratch / "refs.bib")
        shutil.copytree(paths[0].parent / "figures", scratch / "figures")
        key = "full-test-map-mean"
        value = collect(evidence_root)[key]["text"]
        changed = str(Decimal(value) + 1)
        for paper, source in zip(candidates, sources):
            alterations = (
                ("altered paper number", source.replace(r"\ev{" + key + "}{" + value + "}", r"\ev{" + key + "}{" + changed + "}", 1), f"{key}: paper={changed}, source={value}"),
                ("unbound paper number", source.replace(r"\end{document}", "Unbound value 12345.\n" + r"\end{document}"), "unbound numeric literal"),
                ("preamble number macro", source.replace(r"\begin{document}", r"\newcommand{\testmap}{99.99}" + "\n" + r"\begin{document}", 1).replace(r"\begin{abstract}", r"\begin{abstract}" + "\n" + r"\testmap", 1), r"unsupported TeX command: \newcommand"),
                ("evidence macro redefinition", source.replace(r"\begin{document}", r"\renewcommand{\ev}[2]{99}" + "\n" + r"\begin{document}", 1), r"unsupported TeX command: \renewcommand"),
            )
            for label, altered, diagnostic in alterations:
                paper.write_text(altered, encoding="utf-8")
                expect_failure(paper.name + " " + label, candidates, evidence_root, diagnostic)
            paper.write_text(source, encoding="utf-8")
        original_figure = (scratch / "figures/prediction.png").read_bytes()
        shutil.copyfile(scratch / "figures/data_eda.png", scratch / "figures/prediction.png")
        expect_failure("shared figure swap", candidates, evidence_root, "figure differs from generator: figures/prediction.png")
        (scratch / "figures/prediction.png").write_bytes(original_figure)
        if len(candidates) > 1:
            missing = re.sub(PATTERN, "", sources[-1], count=1)
            candidates[-1].write_text(missing, encoding="utf-8")
            expect_failure("translation binding omission", candidates, evidence_root, "paper alignment differs")
            candidates[-1].write_text(sources[-1], encoding="utf-8")
        corrupted_mapping = r"\pdfglyphtounicode{a}{0062}" + "\n" + r"\pdfglyphtounicode{uni1EAD}{0061}" + "\n"
        candidates[-1].write_text(sources[-1].replace(r"\begin{document}", corrupted_mapping + r"\begin{document}", 1), encoding="utf-8")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                compile_paper(source=candidates[-1])
        except ValueError as error:
            if not str(error).startswith("PDF text does not contain source title with Unicode preserved:"):
                raise
            print("EXPECTED FAIL (PDF Unicode mapping corruption): compile rejected")
            print("FAIL:", error)
        else:
            raise ValueError("Corrupted PDF Unicode mapping was accepted")
        candidates[-1].write_text(sources[-1], encoding="utf-8")
        repo = scratch / "repo"
        for relative in {entry["source"] for entry in collect(evidence_root).values()}:
            target = repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(evidence_root / relative, target)
        target = repo / RUN / "test_results.json"
        data = json.loads(target.read_text(encoding="utf-8"))
        data["variants"]["full"]["test"]["map"]["mean"] += 0.01
        target.write_text(json.dumps(data), encoding="utf-8")
        expect_failure("altered source value", candidates, repo, f"{key}: paper={value}, source={changed}")
    print("PASS: alteration proofs used scratch copies; real papers and evidence untouched")
    check(paths, evidence_root)


def expect_failure(label, paths, evidence_root, diagnostic):
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--tex", *map(str, paths), "--evidence-root", str(evidence_root)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, errors="replace")
    failures = [line for line in result.stdout.splitlines() if line.startswith("FAIL:")]
    if result.returncode != 1 or not any(diagnostic in line for line in failures):
        raise ValueError(f"{label}: expected exit 1 with {diagnostic!r}; got exit {result.returncode}: {result.stdout}{result.stderr}")
    print(f"EXPECTED FAIL ({label}): exit {result.returncode}")
    print(next(line for line in failures if diagnostic in line))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", type=Path, nargs="+", default=[ROOT / "main.tex", ROOT / "main_vi.tex"])
    parser.add_argument("--evidence-root", type=Path, default=REPO)
    parser.add_argument("--prove", action="store_true")
    args = parser.parse_args()
    try:
        if args.prove:
            prove(args.tex, args.evidence_root)
        else:
            check(args.tex, args.evidence_root)
    except (ValueError, KeyError, IndexError, OSError) as error:
        print("FAIL:", error)
        sys.exit(1)
