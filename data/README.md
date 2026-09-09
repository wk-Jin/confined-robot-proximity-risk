# Data

Datasets are distributed through GitHub Releases.

Download:

```text
generated_dataset.zip
additional_evaluation_dataset.zip
```

Expected structure after extraction:

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
```

The `layout_image_path` column should contain relative paths such as:

```text
layout_images/layout_000_trial_000.npy
```

The additional evaluation dataset is expected to contain 1,000 samples, with 200 samples in each C0-C4 category.
