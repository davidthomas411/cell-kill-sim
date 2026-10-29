# Lab Record System (`lab/`)

Structured record system for the Cell-Kill-Sim light-sheet CAR-T / Nalm6 reanalysis.

## Roles

- **Technician (`agy`)**: executes versioned protocols, generates real-data galleries, and drafts notebook entries.
- **Postdoc (`claude`)**: reviews every gallery panel, records findings by panel ID, and audits notebook entries and protocols.
- **PI (`user`)**: reviews galleries on key gates and signs off selectively on notebook entries and stage gates (`G0`–`G5`).

## Layout

- `lab/notebook/YYYY-MM-DD-NN-slug.md` — append-only dated lab notebook entries with YAML front matter and six required sections (`Question`, `What was run`, `What was looked at`, `Observations`, `Decision`, `Open issues`).
- `lab/protocols/<id>-v<N>.md` — versioned analytical and review procedures (`gallery-review`, `clean-cell-rule`, `death-calling`, `cup-measurement`).
- `lab/registry.csv` — dataset layer registry with verification status (`unverified`, `verified`, `verified_with_exclusions`, `rejected`) enforced by `tools/lab.py:require_verified`.
- `lab/galleries.csv` — index of real-data visual review galleries.
- `lab/reviews/<gallery_id>.csv` — panel-level review verdicts (`ok`, `problem`, `unsure`), controlled tags, and notes per reviewer.
- `lab/handbook/` — reference documentation on dataset layers (`data.md`), quantitative definitions and caveats (`definitions.md`), known issues linked to notebook entries (`known-issues.md`), and open questions for the Fei lab (`open-questions.md`).
