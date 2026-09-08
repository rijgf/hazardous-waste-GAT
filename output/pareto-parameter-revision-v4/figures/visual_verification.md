# Sensitivity figure variants: visual verification

Verified on 2026-09-08 by opening both generated PNGs with the image-viewing tool and inspecting their displayed content. This is figure-level visual inspection, not a claim to have captured the native Codex Markdown preview.

- Original scatter PNG retained byte-for-byte: `8972c81f2f9bb022a2b59dec1c07b5d7fc0b620847f8b40ab0ece7e10a4326f2`.
- Line PNG inspected: `60dd5da2e0994e1f3643c421ea23d58ad38e8665805631c2f70385190813b634`.
- PCHIP PNG inspected: `aebb51b47b2b6f2e3142c64b3130afec98a55c54450c0a51cc233c4f7f9c6229`.
- Manuscript with all three embedded versions: `c7e183186517bc800549c40a2fb65487146a37c6a6c9b3aee6443b3b1e70d904`.
- Both new images have two readable panels, five identifiable legend entries, intact Chinese labels and axes, no clipped titles, and aligned coordinate scales. Dense hollow markers are omitted in the new versions. The preserved sharp drops remain visible; smoothing does not erase them.
- All three Markdown image links are tested against delivered assets. Captions disclose that segments and interpolated coordinates are visualization aids, not additional feasible solutions. Editable LaTeX formulas are retained; no Word files were modified.
- Full unit suite: 106 tests passed. Independent numerical/report audit: PASS, including 505 original SVG markers, both new curve variants, and unchanged frozen data/model identities.

Earlier `visual-qa/`, `visual-qa-final/`, and `output/markdown-rendering-fix-20260908/` captures remain historical evidence for the manuscript hashes recorded there. They do not validate the current three-figure/LaTeX manuscript.
