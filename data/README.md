# Dataset acquisition and verification

**Current status:** the user explicitly requested a synthetic dataset after the initial data-acquisition blocker. The active source is now the locally generated SpendSense corpus, with its own authored taxonomy. See [synthetic provenance and reproduction](../docs/synthetic_dataset.md). Run `python -m src.generate_synthetic --activate`, then `python -m src.train` to reproduce it. The generator is separate from unit-test fixtures and never inserts rows into the personal ledger.

**Original candidate:** the [AuroraGate challenge data page](https://www.kaggle.com/competitions/aurora-gate-expense-categorization-challenge/data) exposed no readable listing/schema during initial inspection. Its actual file names, labels, license and origin remain unverified. The manual workflow below remains available for acquiring that source or another documented dataset. The active synthetic records do not come from Kaggle.

## Manual download

1. Open the challenge in your own browser. Sign in if required, inspect the rules, permitted uses and available data documentation. Download only if you have permission. Do not put Kaggle credentials in this repository.
2. Place the original downloaded labelled CSV(s) in `data/raw/`. Preserve filenames and original bytes; do not edit them in a spreadsheet. Keep the source README/license alongside them locally. Challenge test files without targets are not suitable for this benchmark.
3. Inspect the actual file, without assuming any column names:

   ```sh
   python -m src.data_loader data/raw/YOUR_FILE.csv
   python -m src.data_loader data/raw/YOUR_FILE.csv --description-column ACTUAL_TEXT_COLUMN --target-column ACTUAL_LABEL_COLUMN
   ```

   This prints filename, SHA-256, shape, headers, missing/blank values and class counts when mapped columns are supplied. It never prints transaction descriptions. CSV must be UTF-8 with a header and comma delimiter; document any necessary conversion as a separate source-preserving step.
4. Copy `data/dataset.example.json` to `data/dataset.json`. Paths in `file` are relative to this manifest. Fill the actual text and target columns, source, license/permission and evidence, source-defined taxonomy and its evidence, and definitions of the mapped fields. Paste the file checksum. Set provenance to `real`, `synthetic`, or `uncertain`, with a factual explanation. Public downloadability does not by itself establish permission or real-world provenance.
5. Set `verified` to `true` only after checking these entries against documentation. This is a human attestation, not an automated proof of truth. The loader checks it, verifies the hash, and rejects labels outside the documented taxonomy. Unknown license blocks training. Uncertain provenance is allowed when explicitly documented and usage permission is established.
6. Run `python -m src.train`, then `python -m src.evaluate`. Check `reports/metrics/<run>/dataset_profile.json` and `models/<run>/metadata.json`. Open the ML Lab for actual results. If cleaning leaves fewer than five unique descriptions per class, obtain more data or a suitable source; never clone records or arbitrarily merge classes to pass validation.

## Another legitimate source

The same manifest supports a different documented **single-label transaction categorization CSV**. Record its original source, permitted uses, label definitions and collection method. Map its description and target explicitly; do not repurpose fraud indicators, sentiment or merchant IDs as expense categories. Multi-label, numeric-only or undocumented-label datasets require a different task and are not supported. No automatic fallback downloads occur. Amount/date columns are ignored by the classifier even if present; the implemented task is description-only.

A source may legitimately publish synthetic labelled transactions; if selected explicitly and documented, mark them synthetic and restrict claims accordingly. Unit-test fixtures in `tests/` are never a source for the real benchmark.

## Files and privacy

`data/processed/<run>/` contains the cleaned text and immutable train/test CSVs. These are local and ignored by Git. Exclusion audits record source CSV row positions and reasons, not descriptions. Conflicting labels remove the entire exact-description group; same-label repeats keep the first. Casefolded, whitespace-normalized exact descriptions are disjoint across train and test. Template overlap diagnostics and a separate grouped evaluation are attempted; these do not identify unseen merchants.

Optional consented campus validation belongs in a separate local directory, never the manifest used for tuning. Collect only anonymized descriptions, expense amounts/ranges and source-defined categories. Keep it out of training and model selection. External-campus validation has not been collected or executed.
