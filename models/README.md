# Models

Trained model checkpoints are distributed through GitHub Releases as:

```text
trained_models.zip
```

The default evaluation and paper-figure scripts expect checkpoints under:

```text
outputs/multi_seed_training/
```

Expected structure:

```text
outputs/multi_seed_training/
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

The checkpoints are included to support exact reproduction of the reported evaluation and Score-CAM figures.
