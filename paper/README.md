# Paper — *When Better Retrieval Doesn't Mean Better Answers*

LNCS-formatted research paper based on the 110-Q evaluation in `eval/results_110/`.

## Files

```
paper/
├── paper.md              # source manuscript (~4,400 words, ~10 LNCS pages)
├── paper.docx            # rendered Word doc, LNCS-styled (built via pandoc)
├── references.bib        # 10 BibTeX entries (numeric [n] LNCS citations)
├── lncs.csl              # Springer LNCS citation style (auto-downloaded)
├── build_docx.sh         # one-line pandoc build script
├── figures/              # 5 figures: arch diagram + 4 charts
└── scripts/              # tiny matplotlib generators for the figures
```

## Edit & rebuild

1. Edit `paper.md` (Markdown — keeps the writing easy to revise).
2. Run `bash build_docx.sh` to regenerate `paper.docx`. Re-runs idempotent and fast.
3. Open `paper.docx` in Word for the final stylistic polish (LNCS-mandated front-page
   widow control, exact font fallbacks, etc., which pandoc handles 90% of).

## Authors and affiliation

The manuscript currently uses placeholders:

```yaml
author:
  - "[Author Name]^[1]^"
  - "[Co-Author 1]^[1]^"
  - "[Co-Author 2]^[1]^"
affiliation:
  - "1: [Department / Institution / Email]"
```

Replace these in `paper.md` (top YAML block) and rebuild.

## Where the numbers come from

Every quantitative claim in the manuscript traces to `eval/results_110/results.json`. The
exact analytic queries are reproduced inline in the methodology and results sections;
numerical values were extracted by the script in `scripts/` (see `make_figures.py` for
metric averages, correlations, partial-coverage counts, etc.).

If you re-run the evaluation and update `eval/results_110/results.json`, regenerate the
figures with:

```bash
/home/archer/Code/LegalAI/eval/.venv/bin/python paper/scripts/make_figures.py
/home/archer/Code/LegalAI/eval/.venv/bin/python paper/scripts/make_architecture.py
bash paper/build_docx.sh
```

## Pandoc note

The build script uses pandoc 3.5 at `~/.local/bin/pandoc` (downloaded as a portable
binary — no system install needed). Override with `PANDOC=/path/to/pandoc bash build_docx.sh`
if you have a different version installed.
