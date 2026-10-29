# Gallery Reviews (`lab/reviews/`)

Append-only CSV logs of panel-by-panel visual reviews, one file per gallery (`<gallery_id>.csv`), written by `tools/review_app.py` or batch review imports.

## Columns

- `panel_id`: panel identifier matching `id` in the gallery's `index.csv`
- `reviewer`: reviewer identifier (`claude`, `user`, etc.)
- `verdict`: `ok` | `problem` | `unsure`
- `tags`: semicolon-delimited list from the fixed vocabulary (`fragmented`, `merged`, `hollow`, `edge`, `tiny`, `misaligned`, `drift`, `label_mismatch`, `neighbour_overlap`, `real_death`, `clean`, `other`)
- `note`: free-text observation for the panel
- `timestamp`: ISO-8601 timestamp when the review row was recorded
