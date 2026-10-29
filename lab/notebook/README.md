# Lab Notebook (`lab/notebook/`)

Append-only dated notebook entries named `YYYY-MM-DD-NN-slug.md`. Create new entries with `python3 tools/lab.py new-entry --title "..." --gate G1`.

## Front Matter Schema (YAML)

- `id`: entry identifier matching filename stem (`YYYY-MM-DD-NN-slug`)
- `date`: `YYYY-MM-DD`
- `title`: short descriptive title
- `gate`: `G0`, `G1`, `G2`, `G3`, `G4`, `G5`, or `none`
- `status`: `draft` | `reviewed` | `signed`
- `author`: author identifier (e.g. `agy`, `claude`, `agy (backfilled)`)
- `reviewers`: list of reviewers
- `signoff`: `who/when` string or `null`
- `code_commit`: git commit hash(es)
- `datasets`: list of `dataset_id`s from `lab/registry.csv`
- `protocols`: list of `id@version` strings from `lab/protocols/`
- `galleries`: list of `gallery_id`s from `lab/galleries.csv`
- `supersedes`: entry ID superseded by this entry, or `null`
- `superseded_by`: entry ID that supersedes this entry, or `null`

## Required Body Sections

1. `## Question`
2. `## What was run`
3. `## What was looked at`
4. `## Observations`
5. `## Decision`
6. `## Open issues`
