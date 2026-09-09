# Release Packaging Guide

This project uses GitHub Releases for large datasets and trained model files.
Do not commit the large `.npy`, `.pt`, or `.zip` files directly to git.

## Required Release Assets

Prepare exactly these three release files:

```text
generated_dataset.zip
additional_evaluation_dataset.zip
trained_models.zip
```

## Source Folders Before Packaging

Place the full local files in this structure before building the release zips:

```text
data/
|-- generated_dataset/
|   |-- trial_summary_all.csv
|   `-- layout_images/
|       |-- layout_000_trial_000.npy
|       `-- ...
`-- additional_evaluation_dataset/
    |-- additional_evaluation_dataset.csv
    `-- layout_images/
        |-- layout_...
        `-- ...

outputs/
`-- multi_seed_training/
    |-- seed_42/
    |   |-- resnet18_layout_risk.pt
    |   `-- vit_tiny_layout_risk.pt
    |-- seed_43/
    |   |-- resnet18_layout_risk.pt
    |   `-- vit_tiny_layout_risk.pt
    `-- seed_44/
        |-- resnet18_layout_risk.pt
        `-- vit_tiny_layout_risk.pt
```

The dataset CSV files should reference images with relative paths such as:

```text
layout_images/layout_000_trial_000.npy
```

## Build Release Archives

From the repository root:

```bash
python scripts/validate_public_release.py
python scripts/build_release_archives.py
```

This creates:

```text
release_assets/
|-- generated_dataset.zip
|-- additional_evaluation_dataset.zip
`-- trained_models.zip
```

## Zip Internal Structure

The two dataset zips include one top-level dataset folder:

```text
generated_dataset.zip
`-- generated_dataset/
    |-- trial_summary_all.csv
    `-- layout_images/

additional_evaluation_dataset.zip
`-- additional_evaluation_dataset/
    |-- additional_evaluation_dataset.csv
    `-- layout_images/
```

The model zip contains seed folders directly:

```text
trained_models.zip
|-- seed_42/
|-- seed_43/
`-- seed_44/
```

This matches the README extraction command:

```bash
unzip trained_models.zip -d outputs/multi_seed_training/
```

