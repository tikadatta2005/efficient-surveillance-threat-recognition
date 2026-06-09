# PyTorch Trainer Framework

## Overview

This module provides a custom training framework built on PyTorch. It supports:

- Training loop abstraction
- Validation loop
- Metrics tracking (TP, TN, FP, FN)
- Accuracy, Precision, Recall, F1-score
- Checkpoint saving
- Configurable logging frequency
- GPU support

It is designed for classification problems (binary or simple multiclass).

---

## Components

### 1. Metrics Class

Tracks evaluation statistics and computes performance metrics.

#### Stored values:
- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)
- Train loss
- Validation loss

#### Methods:

- accuracy()
  -> (TP + TN) / total samples

- precision()
  -> TP / (TP + FP)

- recall()
  -> TP / (TP + FN)

- f1()
  -> Harmonic mean of precision and recall

---

### 2. Trainer Class

Handles full training pipeline.

---

## Initialization

```python
Trainer(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion=None,
    device=None,
    save_dir=None,
    save_checkpoints=True,
    print_every=10
)
```

## Parameters:
* model: torch.nn.Module
* train_loader: DataLoader for training data
* val_loader: DataLoader for validation data
* optimizer: PyTorch optimizer (Adam, SGD, etc.)
* criterion: Loss function (default: CrossEntropyLoss)
* device: "cuda" or "cpu" (auto-detected if None)
* save_dir: directory for checkpoints
* save_checkpoints: enables model saving per epoch
* print_every: logs training loss every N steps

## Training Process
1. Training Phase

For each batch:

* Forward pass
* Compute loss
* Backpropagation (loss.backward)
* Optimizer step

Metrics updated:

* TP, TN, FP, FN
* Training loss

2. Validation Phase

Runs without gradient computation:

* model.eval()
* torch.no_grad()

Metrics updated:

* Validation loss
* TP, TN, FP, FN

### Forward Pass Flow
``` text
x → model → outputs → loss → backward → optimizer.step()
```

### Metrics Computation

Predictions are generated using:
``` py
preds = torch.argmax(outputs, dim=1)
```
Confusion matrix updates:

* TP: predicted 1, actual 1
* TN: predicted 0, actual 0
* FP: predicted 1, actual 0
* FN: predicted 0, actual 1

### Fit Function
``` py
history = trainer.fit(epochs=10)
```
Output:

Returns a list of dictionaries containing:
```py
{
    "epoch": int,
    "train_loss": float,
    "val_loss": float,
    "accuracy": float,
    "precision": float,
    "recall": float,
    "f1": float,
    "tp": int,
    "tn": int,
    "fp": int,
    "fn": int
}
```

### Checkpoint Saving

If enabled:
``` py
save_dir/model_epoch_{epoch}.pt
```
Saved contents:

* model.state_dict()
* optimizer.state_dict()
* epoch number

### Logging

Training loss is printed every:

``` py
print_every
```
Validation and summary metrics are printed after each epoch.

## Limitations
* Designed primarily for classification tasks
* TP/TN/FP/FN logic assumes binary or simplified multiclass
* No built-in support for:
    - segmentation metrics
    - regression metrics
    - distributed training
    - learning rate schedulers (can be added externally)
### Example Usage
``` py
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    save_dir="./checkpoints",
    save_checkpoints=True,
    print_every=20
)

history = trainer.fit(epochs=10)
```

## Design Philosophy

This framework follows PyTorch-native design:

* Model handles forward pass only
* Trainer handles optimization logic
* Metrics handled externally
* Full flexibility for research experimentation