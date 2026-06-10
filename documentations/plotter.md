# plot_training_metrics

## Overview
This function standardizes and visualizes training metrics across epochs. It accepts multiple data formats, converts them into a pandas DataFrame, and generates seaborn line plots for key training and validation metrics.

---

## Function Signature
```python
plot_training_metrics(data, min_value=None)
```

---

## Parameters

### data
Input training history. Supported formats:

1. **dict of lists**
```python
{
    "epoch": [...],
    "train_loss": [...],
    "val_loss": [...],
    ...
}
```

2. **list of dicts**
```python
[
    {"epoch": 1, "train_loss": 0.5, ...},
    {"epoch": 2, "train_loss": 0.4, ...}
]
```

3. **list of lists**
Must follow this column order:
```python
[
    epoch,
    train_loss, val_loss,
    train_accuracy, train_precision, train_recall, train_f1,
    val_accuracy, val_precision, val_recall, val_f1
]
```

---

### min_value (optional)
Dictionary used for filtering returned DataFrame only.
Example:
```python
{"train_loss": 0.8, "val_accuracy": 0.5}
```

Note: This parameter is currently not applied to plotting logic, only intended for post-processing.

---

## Output

### Returns
- `pandas.DataFrame`
  - Sorted by `epoch`
  - Contains normalized training and validation metrics

### Visualization
Generates a single figure with subplots:

1. Loss (train vs validation)
2. Accuracy (train vs validation)
3. Precision (train vs validation)
4. Recall (train vs validation)
5. F1 Score (train vs validation)
6. Empty subplot (reserved)

---

## Dependencies
- pandas
- seaborn
- matplotlib

---

## Behavior Details

### Data Normalization
- Automatically detects input format
- Converts into a unified pandas DataFrame
- Sorts by epoch before plotting

### Plotting
- Uses seaborn `lineplot`
- All metrics plotted as train vs validation pairs
- Uses `darkgrid` style
- Subplots arranged in a 3x2 grid

---

## Example Usage

```python
history = plot_training_metrics({
    "epoch": [1, 2, 3],
    "train_loss": [0.8, 0.6, 0.4],
    "val_loss": [0.9, 0.7, 0.5],
    "train_accuracy": [0.6, 0.7, 0.8],
    "val_accuracy": [0.55, 0.65, 0.75],
    "train_precision": [0.6, 0.7, 0.8],
    "val_precision": [0.5, 0.6, 0.7],
    "train_recall": [0.6, 0.7, 0.8],
    "val_recall": [0.5, 0.6, 0.7],
    "train_f1": [0.6, 0.7, 0.8],
    "val_f1": [0.5, 0.6, 0.7],
})
```

---

## Notes
- Ensure `epoch` column exists in all formats.
- Missing columns may raise pandas errors.
- `axes[5]` is intentionally unused.
