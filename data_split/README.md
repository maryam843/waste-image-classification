# Split file

`trashnet_split.csv` holds the train/val/test assignment for TrashNet glass, metal, paper and plastic.
It contains no images: one row per file with its SHA-256, label, split and name inside the TrashNet zip.

- Built by `waste_cls.data.make_split` (stratified 70/15/15 on unique hashes, seed 42) from the TrashNet zip at commit
  `6fa2b878c6c1b4304b91109070ce0edf9279bb31` (zip SHA-256 `0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2`).
- 1,981 rows: 1,386 train, 297 val, 298 test. Six files are not listed: three duplicate pairs whose copies sit in two
  different class folders.
- `python -m waste_cls.cli prepare-data` rebuilds the split and fails if it differs from this file.
