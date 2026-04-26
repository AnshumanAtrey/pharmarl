# PharmaRL Paper

`AI Alchemy in Medicine: A Vision for LLM-as-Policy Molecular Design via OpenEnv`

## Build

```bash
cd paper
tectonic main.tex
```

Tectonic auto-fetches LaTeX packages and runs biber. First compile takes ~1-2 min for package downloads; subsequent compiles are seconds.

Install tectonic if missing: `brew install tectonic` (macOS) or see https://tectonic-typesetting.github.io.

## Updating empirical results

Single source of truth is the `\newcommand` block at the top of `main.tex`. Update those values once and the whole paper reflects new numbers. The placeholders are:

- `\parseRate` — JSON parse-rate after training (%)
- `\meanReward` / `\maxReward` — group reward statistics
- `\trainSteps` — number of GRPO steps completed
- `\heldOutScore` — held-out target reward
- `\baselineParse` — zero-shot parse rate (control)
- `\wandbRunURL` — link to live W&B run

## Training-curve figure

Place final PNG at `figs/training_curve.png` (overwrite the placeholder). The figure block in `main.tex` already references the file. Recompile to embed.

Suggested figure: 3 panels — mean reward, parse rate, KL-to-reference vs. step. Generate via `examples/plot_results.py` once training W&B run completes.

## Research notes

`research/` contains the literature dumps (4 .md files, ~80 primary citations) used to ground the paper. Not required for compile; useful for review and v2.

## Submission

After empirical numbers are filled in:

1. Recompile: `tectonic main.tex`
2. Verify `main.pdf` is the version you want
3. Upload to Zenodo (https://zenodo.org/deposit/new) → instant DOI
4. Add DOI URL to top-level `README.md` of the repo
