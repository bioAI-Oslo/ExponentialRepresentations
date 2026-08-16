---
name: manuscript
description: Writing style and LaTeX conventions for the exponential map write-ups in tex/ and the markdown analyses in reports/ - scientific prose, hedging, theorem and citation conventions, notation macros, and the constraint that the main paper's source is not in this repository. Use when drafting or editing a write-up, a report, a figure caption or an abstract, when adding a derivation, or when asked to rewrite or restructure paper text.
---

# Manuscript writing

Three kinds of prose live here, and they are held to different standards:

- **`pdfs/Exp_map_spatial_representations.pdf`** — the submitted paper. **Its LaTeX source is not
  in this repository**, only the compiled PDF. You cannot edit the paper from here. When a task
  implies changing it, produce the text and say clearly that it must be carried over by hand;
  never write a `.tex` file in `tex/` and imply the paper has been updated.
- **`tex/`** — standalone write-ups for work not yet in the paper, currently
  `continuum_of_R_writeup.tex`. Full LaTeX, self-contained, intended to be folded into the
  manuscript later.
- **`reports/`** — markdown analyses and discussion. The working record of the exploration strand:
  results, arguments, disagreements about framing. Looser than `tex/`, but the claims in them are
  still claims.

For what the symbols mean and which appendix says what, use `theory-context`. For whether a result
is solid enough to write down, use `model-verification`.

## Voice

- **US English** — "normalized", "generalize", "visualize". Consistent with the paper.
- **Formal and precise**, readable by a computational neuroscientist who does not work on Lie group
  representations. Define terms on first use; expand abbreviations once (CANN, OI, MEC).
- **Hedged where the claim is interpretive, flat where it is derived.** This is the distinction
  that matters most here. An algebraic result is proved, so state it: "commuting generators
  guarantee path independence". A biological reading is inference, so hedge it: "may reflect",
  "is consistent with", "suggests a link". The paper's own phrasing is the model —
  "*Intriguingly*, the average ratio ... falls within the variability range of experimentally
  observed grid cell module spacings, *suggesting* a link".
- **Do not oversell the biology.** The recurring failure mode in the `reports/` drafts is claiming
  a mathematical construction *is* a biological mechanism. Dense R produces mixed-selectivity
  tuning curves; it does not "explain" grid-to-place conversion. Write the mathematical claim and
  the biological analogy as separate sentences.
- **Concise.** No throat-clearing ("It is important to note that"), no restating the previous
  sentence in different words.
- **Punctuation economy.** Avoid stacking colons, semicolons and em-dashes; a colon or dash once or
  twice per paragraph is the ceiling. Stacked dashes read as machine-written.
- **Original.** Never paraphrase closely from a source paper.

## Structure of a derivation

The framework's selling point is transparency, and the prose should show its work in the paper's
order: **state the requirement, derive the algebraic condition, say what it produces, then note the
biological correspondence.**

> To successfully navigate the open field, animals must disambiguate population vectors
> corresponding to distinct locations. → If the generators are skew-symmetric, the exponential is
> orthogonal and the norm is preserved. → Similarity therefore depends only on displacement. →
> This is what CANNs achieve through shifting activity profiles.

Anything derived gets its condition stated explicitly (`[Gx, Gy] = 0`, `Gᵀ = −G`,
`Σ ρ² e^{2iφ} = 0`). Anything empirical gets its parameters stated: `N`, `M`, number of modules,
arena size, bin count, seed.

## LaTeX conventions in `tex/`

Follow `continuum_of_R_writeup.tex`; it is the only exemplar and it is internally consistent.

- **Preamble**: `article`, `amsmath/amssymb/amsthm`, `graphicx`, `booktabs`, `hyperref`,
  `natbib`, `geometry` at 1 in margins.
- **Notation macros** are defined at the top (`\R`, `\pzero`, `\Dir`, `\E`) and used throughout.
  Add a macro rather than respelling a symbol; a bare `\mathbf{R}` in the body is a diff waiting to
  happen.
- **Hard-wrap prose at about 72 columns.** This file is wrapped, not one-sentence-per-line — match
  it. Don't reflow existing paragraphs to a different width in an unrelated edit.
- **Theorems and proofs** use `amsthm` environments with labels
  (`\begin{theorem}[$\R$-Invariance]\label{thm:r-invariance}`). A result that is proved goes in a
  theorem environment; an observation that is not, does not.
- **References** use plain `\ref`/`\eqref` with `thm:`, `sec:`, `eq:`, `fig:` prefixes, and the
  word is written out: `Theorem~\ref{thm:r-invariance}`, `Sec.~\ref{sec:dirichlet}`. There is no
  `cleveref` here — don't introduce it into an existing document.
- **Citations** are `\citep`/`\citet` against an **inline `thebibliography`** at the end of the
  file, with author-year keys (`stensola2012entorhinal`, `sargolini2006conjunctive`). There is no
  `.bib` file in this repository. **Never invent a key**: if a claim needs a reference that is not
  already in the bibliography block, write the claim, flag that an entry is needed, and give enough
  detail to find the paper.
- **Section separators** are the `% ===...===` comment banners around `\section`. Keep the pattern
  when adding a section.
- Cross-referencing the main paper is done in prose (`Sec.~2.3 of the main text`), since the two
  documents do not share a label space.

## Reports

`reports/*.md` are analyses, not polished prose, but they are cited back to and reused, so:

- Open with a **Summary** or **Setup** section that states the configuration —
  `K=16 modules, M=3, N=96, sigma=1.5, seed=...` — before any result. A result without its
  parameters cannot be reproduced or compared against the next run.
- Put numbers in tables with the comparison baseline included. "6.6× MSE improvement" needs the
  two MSEs beside it.
- **Reference figures by their path in `figures/`**, so the claim and the image stay tied together.
- Mark what is established and what is speculation. Several existing reports are explicitly
  critiques of overclaiming — keep that habit; a section headed "Where this overclaims" is more
  useful than a confident summary.
- Report negative and null results. The reason NNLS-for-Gaussian suppressed grid structure is a
  finding, and it is recorded as one.

## Figure captions

Match the paper's style: a **bold title sentence**, then per-panel descriptions keyed `a)`, `b)`,
`c)`, then any parameters needed to read the panel (`N = 300`, `M = 3`, `s = 0` to `s = 5`).
The caption states what is plotted; interpretation belongs in the body.

## Reporting numbers

- **Never write a number you have not been given or computed.** No plausible-looking placeholder
  that could survive into a submission.
- State the precision honestly. `max |Δ| = 1e-6` from a float32 computation is agreement at
  numerical precision — say that, rather than implying exactness the arithmetic cannot support.
- Where a number comes from a stochastic draw, give the seed or report a distribution over draws,
  not a single value presented as if it were deterministic.

## When asked to write a section

1. Read the surrounding text first — match its density, paragraph length and citation rate.
2. Check `theory-context` for notation and appendix letters rather than recalling them.
3. Draft in the existing voice, at the existing wrap width.
4. List separately: claims needing a citation not yet in the bibliography, numbers needing
   computation, and anything you were unsure about. That list is as much the deliverable as the
   prose.
