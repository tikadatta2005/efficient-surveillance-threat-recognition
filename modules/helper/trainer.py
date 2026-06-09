import torch
from torch import nn


class Metrics:
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

        self.train_loss = 0
        self.val_loss = 0

    def accuracy(self):
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0

    def precision(self):
        return self.tp / (self.tp + self.fp + 1e-8)

    def recall(self):
        return self.tp / (self.tp + self.fn + 1e-8)

    def f1(self):
        p = self.precision()
        r = self.recall()
        return 2 * (p * r) / (p + r + 1e-8)


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion=None,
        device=None,
        save_dir=None,
        save_checkpoints=True,
        print_every=10
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer

        self.criterion = criterion or nn.CrossEntropyLoss()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.save_dir = save_dir
        self.save_checkpoints = save_checkpoints
        self.print_every = print_every

    # -----------------------------
    # SAVE CHECKPOINT
    # -----------------------------
    def save(self, epoch):
        if not self.save_checkpoints:
            return

        path = f"{self.save_dir}/model_epoch_{epoch}.pt"

        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, path)

    # -----------------------------
    # METRIC UPDATE (BINARY / MULTICLASS)
    # -----------------------------
    def update_metrics(self, metrics, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)

        metrics.tp += ((preds == 1) & (targets == 1)).sum().item()
        metrics.tn += ((preds == 0) & (targets == 0)).sum().item()
        metrics.fp += ((preds == 1) & (targets == 0)).sum().item()
        metrics.fn += ((preds == 0) & (targets == 1)).sum().item()

    # -----------------------------
    # TRAIN
    # -----------------------------
    def train_one_epoch(self, epoch):
        self.model.train()
        metrics = Metrics()

        total_loss = 0

        for step, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(x)
            loss = self.criterion(outputs, y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # predictions (assumes classification)
            preds = torch.argmax(outputs, dim=1)
            self.update_metrics(metrics, preds, y)

            if step % self.print_every == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

        metrics.train_loss = total_loss / len(self.train_loader)
        return metrics

    # -----------------------------
    # VALIDATION
    # -----------------------------
    def validate(self):
        self.model.eval()
        metrics = Metrics()

        total_loss = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)

                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                self.update_metrics(metrics, preds, y)

        metrics.val_loss = total_loss / len(self.val_loader)
        return metrics

    # -----------------------------
    # FULL TRAIN LOOP
    # -----------------------------
    def fit(self, epochs):
        history = []

        for epoch in range(1, epochs + 1):

            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate()

            self.save(epoch)

            epoch_result = {
                "epoch": epoch,

                "train_loss": train_metrics.train_loss,
                "val_loss": val_metrics.val_loss,

                "accuracy": val_metrics.accuracy(),
                "precision": val_metrics.precision(),
                "recall": val_metrics.recall(),
                "f1": val_metrics.f1(),

                "tp": val_metrics.tp,
                "tn": val_metrics.tn,
                "fp": val_metrics.fp,
                "fn": val_metrics.fn,
            }

            history.append(epoch_result)

            print(
                f"\nEpoch {epoch} | "
                f"Train Loss: {epoch_result['train_loss']:.4f} | "
                f"Val Loss: {epoch_result['val_loss']:.4f} | "
                f"Acc: {epoch_result['accuracy']:.4f} | "
                f"F1: {epoch_result['f1']:.4f}\n"
            )

        return history