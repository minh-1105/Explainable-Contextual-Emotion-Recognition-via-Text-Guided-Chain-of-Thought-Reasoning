# Provenance for `main.tex`

Generated 2026-10-20. Everything here is under `ECER-paper\`. Source folders were only read.
`PROJ` = `E:\Education\FPT Uni\Term5-DSR301m\Explainable Contextual Emotion Recognition via Text-Guided Chain-of-Thought Reasoning\`.
No model was run, nothing was installed, committed or pushed. No local TeX toolchain; compiled on Overleaf (pdfLaTeX + BibTeX, TeX Live 2026): 5 pages, no errors or undefined references.

## 1. Where each number comes from

All numbers in the tables and results text are checked by `scripts\check_numbers.py` (output in `logs\check_numbers_output.txt`, 66 checks, all pass).
The independent re-runs of `PROJ\novelty_evidence\scripts\*.py` were executed with `cwd = PROJ`, `PYTHONDONTWRITEBYTECODE=1`; outputs are in `logs\`.

| Numbers in paper | Source | How obtained |
|---|---|---|
| Label cardinality per split (Table I): people, annotators/person, union and majority labels/person, 1-vote share | `PROJ\Annotations\Annotations.mat` via `votes.py`, `indep.py` | re-run -> `logs\votes.txt`, `logs\indep.txt` |
| Union equals official `combined_categories` for all val/test people | `indep.py` | `logs\indep.txt` (0 mismatches) |
| Majority rule | `votes.py`: label kept if votes >= floor(k/2)+1 for k annotators | code read |
| Empty-majority share 11.7% (val), 19.1% (test) | `check_codex.py` (run with `Annotations\Annotations.mat` as argument) | `logs\check_codex.txt`: 391/3,334 and 1,394/7,280. The literature note says 19.2%; 1,394/7,280 = 19.15%, written as 19.1% |
| Multi-person pairs: 789 images, 2,035 people, 2,017 pairs; 334 (16.6%) disjoint non-empty pairs (val); 996/4,994 = 19.9% (test) | `multi.py`, `pairs_check.py` | `logs\multi.txt`, `logs\pairs_check.txt` |
| ResNet-50 baselines (Table II), train mAP 96.0 / 99.7, val loss minimal at epoch 0 | `..\results\runs\*\*\logs\history.csv`, `config.json` | best row by `val_map`; epochs 1-indexed (19, 4, 25) |
| Pilot setup (300 people, 279 images, seed 42, 409,600 px, 384 tokens, greedy) | `..\results\vlm_pilot\20261019-104614\config.json`, `metrics.json`; model id `Qwen/Qwen2.5-VL-3B-Instruct` | read |
| Pilot D/E macro-F1, micro-F1, predicted labels, parse rate, seconds/sample, truncation (Table III) | `..\results\vlm_pilot\20261019-104614\metrics.json` | read |
| Micro P/R (76.1/21.4, 79.3/25.5), recall by vote count (10/60%, 11/80%; 30 five-vote instances) | `votes.py` | `logs\votes.txt` |
| Macro/micro E-D CIs (union and majority), 2,000-draw paired image bootstrap | `boot.py` | `logs\boot.txt` |
| Engagement counts (120 vs 233 predicted; TP 118 vs 227; support 288), micro-F1 without Engagement 28.3 / 25.4 (-2.9 [-4.8,-1.0]), always-Engagement 26.8 (union) / 46.3 (majority), majority without Engagement -0.2 [-3.8,+3.2] | `engagement.py` | `logs\engagement.txt` |
| Sympathy sensitivity (+2.4 [-0.1,+5.2]) | `macro_sens.py` | `logs\macro_sens.txt` |
| Constant top-k baselines (union: top-2 micro 45.5, top-6 macro 18.1; majority: top-2 micro 52.9, top-6 macro 11.9) | `prior.py`: ranking from validation people excluding the pilot sample | `logs\prior.txt` |
| By-number-of-people macro-F1 and CIs (Table IV) | `..\results\vlm_pilot\20261019-104614\f1_by_people.json` (notebook-side bootstrap) | read |
| Figure 1 | `scripts\make_figure.py` from `history.csv`, `metrics.json`, `logs\engagement.txt`, `logs\prior.txt` | `figures\fig_pilot.pdf/.png` |
| Etesam et al. F1 36.83 / 28.83 / 34.47; "could not employ mAP"; fine-tuned on augmented validation set | `2405.08992v2.pdf` Table I (`pdftotext` -> `logs\etesam2024.txt`) | text read and asserted in `check_numbers.py` |
| Lei et al.: CoT raises precision, lowers recall (EMOTIC); lowers performance on HECO | arXiv 2407.11300 PDF downloaded and extracted -> `logs\lei2024.txt` | asserted in `check_numbers.py` |

**Known discrepancy.** The macro-F1 CI for E-D is [-0.6, +3.0] in `boot.py` (re-run) and [-0.56, +2.92] in `metrics.json` (notebook bootstrap, different random draws). The paper quotes `boot.py` for union macro and micro so that all union/majority intervals come from one script. `check_numbers.py` asserts the two agree within 0.15 points.

**Not comparable on purpose.** Baseline numbers are on all 3,334 validation people with the checkpoint chosen on that set; pilot numbers are on 300 people. The baselines were not run on the pilot sample (that would require running models).

## 2. Statements that rest on the project's literature notes, not on a source I re-read

These come from `PROJ\novelty_evidence\literature\*.md` (which list URLs and quote sources). The paper words them at the level of the abstract/title, and they should be re-checked before submission:
SoV numbered boxes; ELENA structured narratives; Xenos et al. (descriptions -> trained classifier); Ma et al. (token-probability multi-label text); Kim et al. (person-swap consistency); descriptions of ERV, FACR, OPPO, AffectOmni, EDCT-Bench, vCT; the sentence "we found no test that intervenes on person-specific context cues for EMOTIC" (an absence claim from the literature notes, hedged in the paper).

Re-worded on 2026-10-21 after reading the abstracts (arXiv API / Nature page): Martinez-Miwa & Castelan (eight emotion groups, per-image Z-score filter on the votes, not a replacement of vote counting); Hallmen et al. (logit readout, one binary query per category, beats generated answers; thresholding is their control and is worse); DSPO (CVIG masks the image region the reasoning cites and down-weights unsupported interpretations; separate from the DEDR reward); EmoLLM (images and videos, so it is no longer listed under "mostly video with audio").

**Not used because it did not check out.** The literature note quotes Lei et al. as "CoT lowered HECO accuracy 47.22 -> 39.80 at matched 8-shot". The extracted PDF text contains 39.80 (VILA-8B, few-shot CoT) but not 47.22 (the 10-shot non-CoT row reads 47.55). The paper makes only the qualitative statement.

## 3. EmoCoT (Chen & Sun)

Only the Springer abstract page and reference list were read; the full text is paywalled. The paper says so and restricts claims to the abstract. Metadata: Crossref DOI 10.1007/978-981-92-3504-9_49, LNCS vol. 16659, pp. 598-611, ICIC 2026, Crossref issued 2026-07-14. The Springer chapter page gives `citation_publication_date` 2027, © 2027 and cite-as "Chen, Y., Sun, Q. (2027)", publisher Springer, Singapore; the bib uses 2027 (publisher record, which Google Scholar also indexes).

## 4. Other bibliographic caveats

- `kosti2020emotic`: Crossref has the early-access record (pages "1-1", 2019). The bib uses the final TPAMI volume 42(11), pp. 2755-2766, 2020, confirmed by IEEE Xplore (arnumber 8713881) and PubMed 31095475 (Nov 2020, epub 2019-05-14). Author "Jose M. Alvarez" as on Xplore and PubMed (the Crossref deposit drops "M.").
- `lei2025lvlm`: PMLR v260 (ACML 2024 proceedings), published 2025/01/14, so year 2025.
- Published-version check (2026-10-21): every entry was searched for a peer-reviewed version (Google Scholar for the arXiv-cited entries, plus OpenAlex, Crossref, arXiv `comments`, OpenReview, and the PMLR/ICLR/NeurIPS proceedings pages). Entries switched from arXiv to the published version, metadata from the publisher page or DOI record: `etesam2023` (ACII 2024, pp. 10-18), `saim2025elena` (Findings EMNLP 2025), `she2026aica` (Findings ACL 2026), `wu2025customizing` (ICLR 2026, proceedings.iclr.cc pp. 96462-96482), `parcalabescu2024` (ICLR 2025, pp. 21634-21663), `cheng2024emotionllama` (NeurIPS 37), `wei2022cot` (NeurIPS 35), `turpin2023` (NeurIPS 36), `chen2023cs` (ICML 2024, PMLR 235:7880-7904), `wang2026affectomni` (IEEE TAFFC early access; IEEE Xplore arnumber 11585891 on 2026-10-21 still shows volume "PP", issue 99, early-access pages 1-12, so the bib gives no volume/pages), `han2026oppo` (ICML 2026, PMLR 306:40269-40294, published 2026-09-29). `atanasova2023` now carries ACL 2023 Vol. 2 (Short Papers) pp. 283-294 and DOI from Crossref.
- `menzat2026vct`: arXiv comment says accepted to Findings of EMNLP 2026; not yet in the ACL Anthology, so cited as accepted with the arXiv id.
- `lian2026affectgptrl` (replaces `lian2025affectgptr1`, arXiv:2508.01318): the authors' page announces "AffectGPT-R1" accepted in 2026-09, and the NeurIPS 2026 record is poster 149766 "AffectGPT-RL: Revealing Roles of Reinforcement Learning in Open-Vocabulary Emotion Recognition", arXiv:2605.06126, same first author with an extended author list (the poster's list matches arXiv). Treated as the accepted version of the same work; proceedings not yet published, so cited as accepted with the arXiv id.
- Corrections from the 2026-10-21 re-verification (two independent reviewers, then re-checked at the source): `rha2026erv` issue 3 (Crossref); `yang2022heco` LNCS vol. 13697 (Springer); `kim2023vcr` Vol. 1: Long Papers (ACL Anthology); `martinez2023reliability` issue 1, 538 is the article number (Crossref); `luo2020arbee` "Reginald B. Adams Jr." (Springer article page); `cheng2024emotionllama` "Alexander G. Hauptmann", `wei2022cot` "Quoc V. Le", `turpin2023` "Samuel Bowman" (NeurIPS proceedings pages). All IEEE entries were compared with IEEE Xplore (authors, title, pages, year) and match. `etesam2023` keeps "Yal{\c c}{\i}n" (arXiv and Xplore's own IROS 2024 record) although the ACII 2024 Xplore record prints "Yalçin".
- Still arXiv-only, checked 2026-10-21 against every source below; evidence in `logs\arxiv_status.json`: `lanham2023`, `doshivelez2017`, `lian2023emer`, `zhao2025r1omni`, `yang2024emollm`, `bai2025qwen25vl`, `huynh2026facr`, `ding2026edct`, `hallmen2026readout`, `ye2026dspo`. For each: arXiv API has no journal_ref and no DOI; Crossref title search has no exact-title record; OpenAlex lists only arXiv locations; OpenReview (which mirrors DBLP) has only CoRR, except `ye2026dspo` = ICLR 2027 Conference Submission (under review); Google Scholar's "All versions" cluster lists only arXiv, ADS, the authors' own copies (e.g. research.google, anthropic.com) and repositories, no venue. `bai2025qwen25vl` has no Scholar cluster of its own; all Scholar citation records point to arXiv/CoRR. Not finding a record is not proof that none exists; re-run before submission.
- All authors are kept (no "and others").
- `refs.bib` is generated from `logs\ref_sources.json` and `logs\ref_sources_extra.json` by `scripts\make_bib.py`; titles and author counts are re-checked by `scripts\verify_paper.py`.

## 5. Honesty contract, mapped to the text

1. CoT not claimed as novel: Intro para 2 ("not new, and we do not present it as our contribution"); Related Work cites Etesam, Lei, EmoCoT.
2. No claim that E recognizes better: Sec. IV-C "One label explains the micro gain" and "Label policy changes the picture", Fig. 1b/1c. Under majority labels E's macro-F1 (23.7) is above the constant top-6 baseline (11.9); this is reported, not hidden.
3. No "explainable"/faithful claim: abstract, contribution 4 ("research question... no measurement so far"), Limitations paragraph.
4. EmoCoT: abstract only, stated in Related Work.
5. Pilot limits (300 people, one seed, one 3B model, post hoc analyses): Intro, Sec. IV intro, Sec. IV-C setup, Sec. IV-D.

## 6. Commands run (all with `cwd = ECER-paper` unless noted)

```
python scripts/fetch_ref_sources.py      # arXiv (44), Crossref (17), ACL bib, PMLR -> logs/ref_sources.json
python scripts/fetch_ref_extra.py        # Crossref title search for 8 refs -> logs/ref_sources_extra.json
python scripts/make_bib.py               # refs.bib (50 entries)
python scripts/make_figure.py            # figures/fig_pilot.pdf, .png
python scripts/check_numbers.py          # 66 checks
python scripts/verify_paper.py           # cite/bib, structure, word counts
# cwd = PROJ: python novelty_evidence/scripts/{indep,votes,boot,engagement,macro_sens,prior,multi,pairs_check}.py > logs/*.txt
# cwd = PROJ: python novelty_evidence/scripts/check_codex.py Annotations/Annotations.mat > logs/check_codex.txt
```

## 7. Reference verification table

| # | Key | Identifier | Verified through |
|---|-----|------------|------------------|
| 1 | `barrett2019` | DOI 10.1177/1529100619832930 | Crossref record (title, authors, venue, pages) |
| 2 | `barrett2011` | DOI 10.1177/0963721411422522 | Crossref record (title, authors, venue, pages) |
| 3 | `kosti2017emotic` | DOI 10.1109/CVPRW.2017.285 | Crossref record (title, authors, venue, pages) |
| 4 | `kosti2020emotic` | DOI 10.1109/TPAMI.2019.2916866 | Crossref record (title, authors, venue, pages); author names from the version of record (section 4) |
| 5 | `lee2019caer` | DOI 10.1109/iccv.2019.01024 | Crossref title search (exact normalized title + year) |
| 6 | `mittal2020emoticon` | DOI 10.1109/cvpr42600.2020.01424 | Crossref title search (exact normalized title + year) |
| 7 | `yang2022heco` | DOI 10.1007/978-3-031-19836-6_9 | Crossref record (title, authors, venue, pages) |
| 8 | `li2024tscnet` | DOI 10.1145/3664647.3680623 | Crossref record (title, authors, venue, pages) |
| 9 | `yang2024debias` | DOI 10.1109/CVPR52733.2024.01183 | Crossref record (title, authors, venue, pages) |
| 10 | `zhang2023emotionclip` | DOI 10.1109/cvpr52729.2023.01821 | Crossref title search (exact normalized title + year) |
| 11 | `etesam2023` | arXiv:2310.19995 | arXiv API record (title, authors); venue, pages, DOI from the published version (section 4) |
| 12 | `etesam2024` | DOI 10.1109/IROS58592.2024.10802538 | Crossref record (title, authors, venue, pages); full PDF read |
| 13 | `lei2025lvlm` | arXiv:2407.11300 | arXiv API record (title, authors); venue, pages, DOI from the published version (section 4); PMLR page metadata (v260, pp. 111-126, 2025/01/14); full PDF read (CoT statements) |
| 14 | `bhattacharyya2025eve` | DOI 10.18653/v1/2025.findings-naacl.97 | Crossref record (title, authors, venue, pages); ACL Anthology .bib |
| 15 | `xenos2025vllm` | DOI 10.1109/IJCNN64981.2025.11227260 | Crossref record (title, authors, venue, pages) |
| 16 | `chen2026emocot` | DOI 10.1007/978-981-92-3504-9_49 | Crossref record (title, authors, venue, pages); Springer page (abstract and reference list only; full text paywalled) |
| 17 | `lanham2023` | arXiv:2307.13702 | arXiv API record (title, authors, date); cited as arXiv preprint; no published version found (section 4) |
| 18 | `jacovi2020` | DOI 10.18653/v1/2020.acl-main.386 | Crossref record (title, authors, venue, pages); ACL Anthology .bib |
| 19 | `luo2020arbee` | DOI 10.1007/s11263-019-01215-y | Crossref record (title, authors, venue, pages); author names from the version of record (section 4) |
| 20 | `he2016resnet` | DOI 10.1109/cvpr.2016.90 | Crossref title search (exact normalized title + year) |
| 21 | `yang2023captions` | DOI 10.1109/acii59096.2023.10388198 | Crossref title search (exact normalized title + year) |
| 22 | `zhang2024sov` | DOI 10.18653/v1/2024.emnlp-main.257 | Crossref record (title, authors, venue, pages); ACL Anthology .bib |
| 23 | `saim2025elena` | arXiv:2509.19595 | arXiv API record (title, authors); venue, pages, DOI from the published version (section 4) |
| 24 | `she2026aica` | arXiv:2604.05900 | arXiv API record (title, authors); venue, pages, DOI from the published version (section 4) |
| 25 | `wu2025customizing` | arXiv:2509.21950 | arXiv API record (title, authors); venue, pages, DOI from the published version (section 4) |
| 26 | `lin2025eibench` | DOI 10.1109/cvprw67362.2025.00515 | Crossref title search (exact normalized title + year) |
| 27 | `lian2023emer` | arXiv:2306.15401 | arXiv API record (title, authors, date); cited as arXiv preprint; no published version found (section 4) |
| 28 | `cheng2024emotionllama` | arXiv:2406.11161 | arXiv API record (title, authors); venue, pages, DOI from the published version (section 4); author names from the version of record (section 4) |
| 29 | `zhao2025r1omni` | arXiv:2503.05379 | arXiv API record (title, authors, date); cited as arXiv preprint; no published version found (section 4) |
| 30 | `lian2026affectgptrl` | arXiv:2605.06126 | arXiv API record (title, authors, date); accepted, proceedings not yet published; cited as accepted (section 4) |
| 31 | `yang2024emollm` | arXiv:2406.16442 | arXiv API record (title, authors, date); cited as arXiv preprint; no published version found (section 4) |
| 32 | `wei2022cot` | arXiv:2201.11903 | arXiv API record (title, authors); venue, pages, DOI from the published version (section 4); author names from the version of record (section 4) |
| 33 | `turpin2023` | arXiv:2305.04388 | arXiv API record (title, authors); venue, pages, DOI from the published version (section 4); author names from the version of record (section 4) |
| 34 | `doshivelez2017` | arXiv:1702.08608 | arXiv API record (title, authors, date); cited as arXiv preprint; no published version found (section 4) |
| 35 | `hase2020las` | DOI 10.18653/v1/2020.findings-emnlp.390 | Crossref title search (exact normalized title + year) |
| 36 | `chen2023cs` | arXiv:2307.08678 | arXiv API record (title, authors); venue, pages, DOI from the published version (section 4) |
| 37 | `atanasova2023` | arXiv:2305.18029 | arXiv API record (title, authors); venue, pages, DOI from the published version (section 4) |
| 38 | `parcalabescu2024` | arXiv:2404.18624 | arXiv API record (title, authors); venue, pages, DOI from the published version (section 4) |
| 39 | `ding2026edct` | arXiv:2609.17953 | arXiv API record (title, authors, date); cited as arXiv preprint; no published version found (section 4) |
| 40 | `menzat2026vct` | arXiv:2609.06704 | arXiv API record (title, authors, date); accepted, proceedings not yet published; cited as accepted (section 4) |
| 41 | `rha2026erv` | DOI 10.1609/aaai.v40i3.37184 | Crossref title search (exact normalized title + year) |
| 42 | `huynh2026facr` | arXiv:2606.15779 | arXiv API record (title, authors, date); cited as arXiv preprint; no published version found (section 4) |
| 43 | `han2026oppo` | arXiv:2606.25325 | arXiv API record (title, authors); venue, pages, DOI from the published version (section 4) |
| 44 | `ye2026dspo` | arXiv:2609.36775 | arXiv API record (title, authors, date); cited as arXiv preprint; no published version found (section 4) |
| 45 | `wang2026affectomni` | arXiv:2608.26193 | arXiv API record (title, authors); venue, pages, DOI from the published version (section 4) |
| 46 | `martinez2023reliability` | DOI 10.1038/s41597-023-02435-1 | Crossref record (title, authors, venue, pages) |
| 47 | `ma2025mlc` | DOI 10.18653/v1/2025.emnlp-main.126 | Crossref record (title, authors, venue, pages); ACL Anthology .bib |
| 48 | `hallmen2026readout` | arXiv:2610.08162 | arXiv API record (title, authors, date); cited as arXiv preprint; no published version found (section 4) |
| 49 | `kim2023vcr` | DOI 10.18653/v1/2023.ijcnlp-main.66 | Crossref record (title, authors, venue, pages); ACL Anthology .bib |
| 50 | `bai2025qwen25vl` | arXiv:2502.13923 | arXiv API record (title, authors, date); cited as arXiv preprint; no published version found (section 4) |

Total cited references: 50
