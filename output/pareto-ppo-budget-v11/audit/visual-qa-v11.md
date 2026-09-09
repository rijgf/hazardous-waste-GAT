# V9 12000-candidate report visual and Markdown checks

The root agent visually inspected the three generated PNGs with the image viewer. Both panels, Chinese titles and axis labels, units, five-level legends, colors and line styles are legible and are not clipped. The dense scatter version is retained alongside line and PCHIP smoothing versions. Curves are visual guides, not additional feasible solutions; raw non-dominated points remain the data source.

Inspected PNG SHA256 values:

| Figure | SHA256 |
| --- | --- |
| sensitivity.png | c6f44e78af62581fe88c2c714daf05a73bee7bc045af563c03babefd2ad32611 |
| sensitivity_line.png | ee3f3a4ea3630d8396d6c643c29deb525d8781396459afd8d646cade9cd2fb23 |
| sensitivity_smooth.png | b7cdb3dad64242a1c3c5832a6064ee37cceda4c71995792978c63ab095594ed3 |

The staged Markdown contains seven structurally consistent GFM tables (data rows x columns: 10x8, 1x5, 2x9, 2x5, 13x7, 9x10, 2x6), three correctly paired display-math blocks and three figure links. All three editable LaTeX blocks are byte-identical in content to the previous formal report; no formula image was introduced.

Scope: actual PNG visual inspection and Markdown structural checks. This is not a claim that the current document was visually inspected inside Codex's native Markdown renderer. Final publication must recheck these PNG hashes and the published document structure, because report publication regenerates figure files.

Final publication recheck: all three PNG hashes above remain identical. Published manuscript SHA256 is `198666d3bdb2b55b1ce4e69babca519533f8cbf8f391509ddf68bd01776a9324`; report_verification.json SHA256 is `00ed43152b43df7759cdaeb9627c9768d8a579cbb08e9cf22b9e0d7508fc23bf`. Editable mathematics remain identical to the pre-run backup. The root link parser matched 20 existing local targets; the independent audit separately checks its full parsed link inventory. SVG metadata and variant provenance were regenerated during publication and must use their final hashes, not the earlier staged SVG hashes.
