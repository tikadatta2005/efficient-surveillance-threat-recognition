import torch


class Metrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes

        self.confusion_matrix = torch.zeros(
            (num_classes, num_classes),
            dtype=torch.long
        )

        self.train_loss = 0
        self.val_loss = 0

    def update(self, preds, targets):
        preds = preds.view(-1).cpu()
        targets = targets.view(-1).cpu()

        for t, p in zip(targets, preds):
            self.confusion_matrix[t, p] += 1

    def accuracy(self):
        cm = self.confusion_matrix.float()
        correct = torch.diag(cm).sum()
        total = cm.sum()

        return (correct / total).item() if total > 0 else 0

    def precision(self):
        cm = self.confusion_matrix.float()

        precisions = []

        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp

            precisions.append(tp / (tp + fp + 1e-8))

        return torch.mean(torch.stack(precisions)).item()

    def recall(self):
        cm = self.confusion_matrix.float()

        recalls = []

        for c in range(self.num_classes):
            tp = cm[c, c]
            fn = cm[c, :].sum() - tp

            recalls.append(tp / (tp + fn + 1e-8))

        return torch.mean(torch.stack(recalls)).item()

    def f1(self):
        cm = self.confusion_matrix.float()

        f1s = []

        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)

            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            f1s.append(f1)

        return torch.mean(torch.stack(f1s)).item()