# Lab Protocols (`lab/protocols/`)

Versioned analysis and review procedures named `<id>-v<N>.md`.

## Front Matter Schema (YAML)

- `id`: protocol identifier (`gallery-review`, `clean-cell-rule`, `death-calling`, `cup-measurement`)
- `version`: version tag (`v1`, `v2`, ...)
- `date`: `YYYY-MM-DD`
- `status`: `draft` | `reviewed` | `signed`

## Active & Draft Protocols

- `gallery-review-v1.md` — panel-by-panel visual verification protocol, verdict scale, tag vocabulary, sampling design, and dual-reviewer gate rule.
- `clean-cell-rule-v1.md` — Gate G1 pre-outcome mask quality criteria for single, whole, un-truncated cells.
- `death-calling-v1.md` — Gate G2 target cell death calling procedure from 2D 405-nm SYTOX projections and 3D contacted-target instances (`src/death_labels.py`).
- `cup-measurement-v1.md` — Gate G3 3D synaptic contact patch and convex-hull / spherical indentation measurement procedure (`src/contact_descriptors.py`).
