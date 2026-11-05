---
name: manuscript
description: Writing style and LaTeX conventions for the exponential map manuscript and write-ups around it - scientific prose, hedging, derivation structure, citations, captions, reporting numbers, and the constraint that the manuscript is edited elsewhere and never from this repository. Use when drafting or editing manuscript text, a figure caption, an abstract or a write-up, when adding a derivation, or when asked to rewrite or restructure paper text.
---

# Manuscript writing

The paper is not edited from this repository.

- **`pdfs/Exp_map_spatial_representations.pdf`** — the submitted paper, compiled.
- **`paper/`** — a mirror of the manuscript's LaTeX source, present in a working checkout. The
  manuscript itself is edited elsewhere and the mirror is synced by hand, so it can lag. **Read
  it, never edit it.** When a task implies changing the paper, produce the text and say that it
  must be carried over by hand: a short block naming the passage to replace and its replacement,
  one or two at a time, in document order. Anchor each block on text that is actually in the
  mirror, and ask for a fresh sync before anchoring on a passage that has been under revision.
- **Notebook markdown, READMEs and write-ups** — looser than the paper, but the claims in them are
  still claims, and the rules below on hedging and numbers apply.

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
- **Do not oversell the biology.** The recurring failure mode is claiming a mathematical
  construction *is* a biological mechanism. A fitted model that grows further fields beyond the
  arena *predicts* multiple fields in larger environments; it does not show that place cells work
  this way. Write the mathematical claim and the biological analogy as separate sentences.
- **Accommodates, not validates.** Where the framework reproduces a known finding, say what it
  accommodates or is consistent with. Avoid "prove", "validate", "confirm", "require",
  "strictly" and "perfectly" for anything that is not an algebraic identity.
- **Model output is "place-like fields"**, never "place fields"; roots of unity are "a solution",
  not "the canonical solution".
- **Concise.** No throat-clearing ("It is important to note that"), no restating the previous
  sentence in different words.
- **One claim per sentence**, in the article's register ("Furthermore", "Crucially", "In
  contrast"). Watch for the same verb carrying three sentences in a row.
- **Punctuation economy.** In manuscript prose, no em-dashes, semicolons or colons; the colon
  after a panel label in a caption is the exception. Elsewhere a colon or dash once or twice per
  paragraph is the ceiling. Stacked dashes read as machine-written.
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

## LaTeX conventions

Match the manuscript; `paper/main.tex` is the exemplar when it is present.

- **No theorem environments.** The paper derives in the physics style, in flowing prose with
  displayed equations: state the problem, lay out the conditions, derive. Do not present a
  result as a theorem, lemma or corollary, and do not call a finding a "theorem" in prose; say
  "result", "condition" or "characterization". Introduce an unfamiliar technical notion in one
  plain sentence at first use.
- **References** use `cleveref`: `\cref{eq:cosine_similarity_final}`, `\cref{app:experimental_data}`.
  Reuse the labels that exist rather than inventing parallel ones.
- **Citations** are `\citep`/`\cite` against `references.bib`, with keys of the form
  `gardner_toroidal_2022`. **Never invent a key and never cite from memory**: check every
  citation, and every claim about what a paper found, against the source (the DOI record, the
  paper itself) before proposing it, and say what was checked. If a claim needs a reference that
  is not in the bibliography, write the claim, flag that an entry is needed, and give enough
  detail to find the paper.
- **Do not reflow** existing paragraphs to a different width in an unrelated edit.

## Figure captions

Match the paper's style: a **bold title sentence**, then per-panel descriptions keyed `a)`, `b)`,
`c)`, then any parameters needed to read the panel (`N = 300`, `M = 3`, `s = 0` to `s = 5`).
The caption states what is plotted; interpretation belongs in the body.

## Reporting numbers

- **Never write a number you have not been given or computed.** No plausible-looking placeholder
  that could survive into a submission.
- The manuscript carries the narrative, not the bookkeeping. Sweep tables, noise ceilings and
  residuals of checks belong in the notebooks; the text gets the number that makes the point.
- State the precision honestly. `max |Δ| = 1e-6` from a float32 computation is agreement at
  numerical precision — say that, rather than implying exactness the arithmetic cannot support.
- Where a number comes from a stochastic draw, give the seed or report a distribution over draws,
  not a single value presented as if it were deterministic.

## When asked to write a section

1. Read the surrounding text first — match its density, paragraph length and citation rate.
2. Check `theory-context` for notation and appendix letters rather than recalling them.
3. Draft in the existing voice, at the existing wrap width, as blocks to paste rather than edits
   to the source.
4. List separately: claims needing a citation not yet in the bibliography, numbers needing
   computation, and anything you were unsure about. That list is as much the deliverable as the
   prose.
