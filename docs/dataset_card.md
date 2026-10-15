# Dataset card

**Active source: explicitly synthetic demonstration data.** Generated locally at the user's request. No real customer/bank records or external dataset was used.

| Field | Verified local value |
|---|---|
| File | `spendsense_synthetic_v1_seed42.csv` |
| Records / columns | 2320 / 2 |
| Input / target | `description` / `category` |
| Missing values | `{"description": 0, "category": 0}` |
| Blank values | `{"description": 0, "category": 0}` |
| Retained after cleaning | 2320 |
| Exclusion counts | `{}` |
| Train / primary test | 1856 / 464 |
| Seed / generator version | 42 / 1 |
| SHA-256 | `7d22fbc094b52c1aea83822b740eee59a28d44924667a22de8990f64a707c1b9` |
| Source | `local:src/generate_synthetic.py` |
| Usage record | Project-authored synthetic demonstration data; authorized for this project |
| Run | `20261010T121905Z-f76a69e2` |

## Observed class distribution

| Category | Records |
|---|---:|
| Food and Dining | 360 |
| Transportation | 320 |
| Groceries | 280 |
| Shopping | 240 |
| Education | 240 |
| Bills and Utilities | 200 |
| Entertainment | 180 |
| Subscriptions | 180 |
| Travel | 160 |
| Health | 160 |

These counts were read from the executed run metadata, not estimated. Class sizes are generator design choices, not observed student prevalence. Only description text is a feature.

The exact-description split has zero identical normalized descriptions across partitions, but 349 template groups overlap. The independent grouped evaluation status is **completed**. It removes that proxy overlap; both evaluations still share the same authored vocabulary and generation recipe.

The manifest records checksum, seed, generator hash, definitions and provenance. Its verified flag confirms the locally authored schema/recipe, not independent human annotation or real-world quality. Review [the generation protocol and full taxonomy](synthetic_dataset.md) for ambiguity and limitations.

The original AuroraGate source remains unverified and is not used. Real campus validation has not been collected. A legitimate alternate source can be imported using `data/README.md`.
