---
name: reading-papers
description: How to read the PDFs in pdfs/ and other reference material using pdftotext rather than opening the binary directly. Use when asked to read, summarise, extract a derivation from, or cite a PDF, when checking what our own manuscript actually says, or when comparing our framework against a published approach.
---

# Reading PDFs

PDFs are read with `pdftotext` from the `poppler` package, installed at
`/opt/homebrew/bin/pdftotext`.

```bash
# Convert to text in the scratchpad, not the repository
pdftotext pdfs/Exp_map_spatial_representations.pdf /tmp/paper.txt
wc -l /tmp/paper.txt

# Locate the sections you need before reading
grep -n "Abstract\|Introduction\|Results\|Conclusion\|References" /tmp/paper.txt
```

Then read the text file in chunks, starting from the section offsets rather than page one.

- **Never** open a `.pdf` with the Read tool — it returns binary noise.
- Write the `.txt` to the scratchpad or `/tmp`, never into the repository working tree.
- `pdftotext` handles digitally-produced PDFs reliably. It fails on scanned or image-only
  documents; if the output is empty or garbled, say so rather than guessing at the contents.
- `pdftotext -layout` preserves the visual arrangement and reads better for tables, equations and
  figure captions. **Use it for our own manuscript**: the NeurIPS submission carries line numbers
  in the margin, and without `-layout` they interleave with the body text and make the first
  hundred lines unreadable.

## What is in `pdfs/`

- `Exp_map_spatial_representations.pdf` — our manuscript, the NeurIPS submission. The source
  of truth for what the framework claims and for the section and appendix numbering used
  throughout the code. `theory-context` carries a map of its sections and appendices; when you
  need the actual derivation, come here.
- `Constraining R Matrix for Grid Cells.pdf` — reference material for the R exploration strand.

The appendices run A through L and carry most of the derivations; section numbering in the body
stops at 5. Appendix headings sit under the NeurIPS checklist in the text dump, well past the
references, so grep for the heading rather than scrolling.

## Reading for a purpose

Say why you are reading before you start, and read only that far.

- **A derivation the code implements:** go to the appendix named in the docstring and check that
  the code's construction matches the algebra step for step — the eigenvalue convention, the
  direction of the conjugation, the normalisation. This is the highest-value use of the paper.
- **A specific claim:** find the result in the text, then check the figure and the parameters
  behind it. What a paper asserts in its abstract and what its figures show under a stated `N` and
  `M` are not always the same.
- **Methods for comparison:** extract the concrete parameters — `N`, symmetry order, eigenvalue
  scale, arena size, bin count, number of training steps for the Appendix K fit — and note where
  they differ from ours.
- **Citing:** confirm the claim is actually in the paper and is the paper's own finding rather than
  its summary of someone else's. Our manuscript cites heavily; a result attributed to Issa & Zhang
  or Xu et al. is theirs, not ours.

Quote or paraphrase precisely, and never state a number, parameter, equation or appendix letter as
being in a paper without having read that passage. If the PDF does not contain what was expected,
report that.
