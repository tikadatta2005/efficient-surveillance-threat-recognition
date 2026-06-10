TESTER CLASS DOCUMENTATION
==========================

Overview
--------
The Tester class is used to evaluate a trained PyTorch model on a test dataset.
It computes loss and multiple classification metrics without performing any training or weight updates.

It is designed to be consistent with the Trainer class and uses the same Metrics utility.

------------------------------------------------------------

Class: Tester
------------------------------------------------------------

Purpose:
- Evaluate model performance on test data
- Compute loss and classification metrics
- Optionally return predictions and ground truth labels

------------------------------------------------------------

Initialization
------------------------------------------------------------

Tester(
    model,
    test_loader,
    num_classes,
    criterion=None,
    device=None
)

Parameters:

model:
    PyTorch model to evaluate.

test_loader:
    DataLoader providing test dataset batches.

num_classes:
    Number of target classes used for Metrics initialization.

criterion (optional):
    Loss function used for evaluation.
    Default: nn.CrossEntropyLoss()

device (optional):
    Device for computation ("cuda" or "cpu").
    Default: CUDA if available, else CPU.

------------------------------------------------------------

Method: test
------------------------------------------------------------

test(return_predictions=False)

Description:
Runs evaluation on the test dataset and computes:
- Loss
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Parameters:

return_predictions (bool):
    If True, returns model predictions and ground truth labels.

------------------------------------------------------------

Execution Flow
------------------------------------------------------------

1. Model is set to evaluation mode:
   model.eval()

2. Gradients are disabled:
   with torch.no_grad()

3. For each batch:
   - Move data to device
   - Perform forward pass
   - Compute loss
   - Extract predictions using argmax
   - Update Metrics object

4. Aggregate results:
   - Average test loss computed over all batches
   - Metrics computed via Metrics class

------------------------------------------------------------

Returned Output
------------------------------------------------------------

A dictionary containing:

test_loss:
    Average loss over test dataset.

test_accuracy:
    Accuracy computed on test data.

test_precision:
    Precision score (from Metrics class).

test_recall:
    Recall score (from Metrics class).

test_f1:
    F1 score (from Metrics class).

confusion_matrix:
    Confusion matrix tensor (CPU, detached).

Optional (if return_predictions=True):

predictions:
    Tensor of all predicted labels.

targets:
    Tensor of all ground truth labels.

------------------------------------------------------------

Dependencies
------------------------------------------------------------

- torch
- torch.nn
- modules.helper.Metrics

------------------------------------------------------------

Notes
------------------------------------------------------------

- No training or optimization is performed.
- Fully deterministic evaluation (no gradients).
- Designed to match Trainer metric structure.
- Assumes Metrics class provides:
    accuracy(), precision(), recall(), f1(), update(), confusion_matrix

------------------------------------------------------------
END OF DOCUMENTATION
------------------------------------------------------------