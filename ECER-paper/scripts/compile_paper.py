import argparse
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIKTEX = Path(os.environ["LOCALAPPDATA"]) / "Programs/MiKTeX/miktex/bin/x64"


def compile_paper(source=ROOT / "main.tex", build=None):
    build = Path(tempfile.mkdtemp(prefix=f"ecer-{source.stem}-")) if build is None else build.resolve()
    if build.is_relative_to(ROOT.parent.resolve()):
        raise ValueError("Compile directory must be outside the repository")
    print(f"Compiling {source.name}", flush=True)
    build.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, build / "main.tex")
    shutil.copyfile(source.parent / "refs.bib", build / "refs.bib")
    shutil.copytree(source.parent / "figures", build / "figures", dirs_exist_ok=True)
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join((str(MIKTEX), r"C:\Windows\System32", r"C:\Windows"))
    latex = [str(MIKTEX / "pdflatex.exe"), "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "main.tex"]
    commands = [latex, [str(MIKTEX / "bibtex.exe"), "main"], latex, latex]
    for index, command in enumerate(commands):
        result = subprocess.run(command, cwd=build, env=environment, capture_output=True, text=True, errors="replace")
        (build / f"pass-{index}.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        print(f"{Path(command[0]).name} pass {index}: exit {result.returncode}", flush=True)
        if result.returncode:
            raise ValueError(f"Compilation failed in {build}:\n" + (result.stdout + result.stderr)[-5000:])
    log = (build / "main.log").read_text(encoding="utf-8", errors="replace")
    bibliography_log = (build / "main.blg").read_text(encoding="utf-8", errors="replace")
    unresolved = [line for line in log.splitlines() if re.search(r"undefined|multiply defined|Label\(s\) may have changed", line, re.I)]
    warnings = [line for line in bibliography_log.splitlines() if "Warning--" in line]
    overfull = [float(value) for value in re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", log)]
    page_match = re.search(r"Output written on main.pdf \((\d+) pages?", log)
    if page_match is None:
        raise ValueError("No page count in successful compilation log")
    pages = int(page_match[1])
    if pages < 4 or unresolved or warnings or any(width > 10 for width in overfull):
        raise ValueError(f"Compile requirements failed: pages={pages}, unresolved={unresolved}, bibliography warnings={warnings}, overfull widths={overfull}; build={build}")
    title = re.search(r"\\title\{([^{}]+)\}", source.read_text(encoding="utf-8-sig"))
    if title is None:
        raise ValueError("No plain-text title in paper source")
    extraction = subprocess.run(
        [str(MIKTEX / "pdftotext.exe"), "-f", "1", "-l", "1", "-enc", "UTF-8", "main.pdf", "-"],
        cwd=build, env=environment, capture_output=True, encoding="utf-8", errors="replace",
    )
    if extraction.returncode:
        raise ValueError(f"PDF text extraction failed in {build}: {extraction.stderr}")
    expected_title = " ".join(unicodedata.normalize("NFC", title[1]).split())
    extracted_text = " ".join(unicodedata.normalize("NFC", extraction.stdout).split())
    if expected_title not in extracted_text:
        raise ValueError(f"PDF text does not contain source title with Unicode preserved: {source.name}; build={build}")
    print(f"PASS: real MiKTeX compile; {pages} pages; no undefined citations/references; {len(overfull)} overfull boxes, none >10pt")
    print("PASS: pdftotext extraction contains source title with Unicode preserved")
    print(f"Build directory: {build}")
    return build


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path)
    args = parser.parse_args()
    compile_paper(build=args.build_dir)
