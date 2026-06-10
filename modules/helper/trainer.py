import torch
from torch import nn
from modules.helper.Metrics import Metrics
import os

class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        num_classes,
        criterion=None,
        device=None,
        save_dir=None,
        save_checkpoints=None,
        print_every=10
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer

        self.num_classes = num_classes

        self.criterion = criterion or nn.CrossEntropyLoss()

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)

        self.save_dir = save_dir
        self.save_checkpoints = save_checkpoints
        self.print_every = print_every

    # ----------------------------------
    # SAVE
    # ----------------------------------

    def save(self, epoch):
        
        if self.save_dir is None:
            return
        
        if not isinstance(self.save_checkpoints, int):
            return

        if self.save_checkpoints <= 0:
            return

        if epoch % self.save_checkpoints != 0:
            return
        
        os.makedirs(self.save_dir, exist_ok=True)

        path = f"{self.save_dir}/model_epoch_{epoch}.pt"

        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()
            },
            path
        )

    # ----------------------------------
    # TRAIN ONE EPOCH
    # ----------------------------------

    def train_one_epoch(self, epoch):

        self.model.train()

        metrics = Metrics(self.num_classes)

        total_loss = 0

        for step, (x, y) in enumerate(self.train_loader):

            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(x)

            loss = self.criterion(outputs, y)

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            metrics.update(preds, y)

        metrics.train_loss = total_loss / len(self.train_loader)

        return metrics

    # ----------------------------------
    # VALIDATION
    # ----------------------------------

    def validate(self):

        self.model.eval()

        metrics = Metrics(self.num_classes)

        total_loss = 0

        with torch.no_grad():

            for x, y in self.val_loader:

                x = x.to(self.device)
                y = y.to(self.device)

                outputs = self.model(x)

                loss = self.criterion(outputs, y)

                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)

                metrics.update(preds, y)

        metrics.val_loss = total_loss / len(self.val_loader)

        return metrics

    # ----------------------------------
    # FIT
    # ----------------------------------

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

                "train_accuracy": train_metrics.accuracy(),
                "train_precision": train_metrics.precision(),
                "train_recall": train_metrics.recall(),
                "train_f1": train_metrics.f1(),

                "val_accuracy": val_metrics.accuracy(),
                "val_precision": val_metrics.precision(),
                "val_recall": val_metrics.recall(),
                "val_f1": val_metrics.f1(),

                "confusion_matrix":
                    val_metrics.confusion_matrix.clone()
            }

            history.append(epoch_result)

            print(
                f"Epoch [{epoch}/{epochs}] | "
                f"Train Loss: {epoch_result['train_loss']:.4f} | "
                f"Val Loss: {epoch_result['val_loss']:.4f} | "
                f"Train Acc: {epoch_result['train_accuracy']:.4f} | "
                f"Val Acc: {epoch_result['val_accuracy']:.4f} | "
                f"Train F1: {epoch_result['train_f1']:.4f} | "
                f"Val F1: {epoch_result['val_f1']:.4f}"
            )

        return history