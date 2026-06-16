import os
import re
import copy
import torch
from torch import nn
from modules.helper.Metrics import Metrics


class Tester:
    def __init__(
        self,
        model,
        test_loader,
        num_classes,
        criterion=None,
        device=None
    ):
        self.model = model
        self.test_loader = test_loader

        self.num_classes = num_classes
        self.criterion = criterion or nn.CrossEntropyLoss()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    # ----------------------------------
    # TEST
    # ----------------------------------

    def test(self, return_predictions=False):
        self.model.eval()

        metrics = Metrics(self.num_classes)
        total_loss = 0.0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                metrics.update(preds, y)

                if return_predictions:
                    all_preds.append(preds.detach().cpu())
                    all_targets.append(y.detach().cpu())

        metrics.loss = total_loss / len(self.test_loader)

        if hasattr(metrics, "compute"):
            metrics.compute()

        result = {
            "test_loss": metrics.loss,
            "test_accuracy": metrics.accuracy(),
            "test_precision": metrics.precision(),
            "test_recall": metrics.recall(),
            "test_f1": metrics.f1(),

            "confusion_matrix": (
                metrics.confusion_matrix.detach().cpu().clone()
                if torch.is_tensor(metrics.confusion_matrix)
                else metrics.confusion_matrix.copy()
            )
        }

        if return_predictions:
            result["predictions"] = torch.cat(all_preds) if all_preds else None
            result["targets"] = torch.cat(all_targets) if all_targets else None

        return result


    # ----------------------------------
    # TEST ALL CHECKPOINTS
    # ----------------------------------

    def test_all_checkpoints(self, model_dir, return_predictions=True):

        results = []

        model_files = [
            f for f in os.listdir(model_dir)
            if f.endswith(".pt") and "epoch_" in f
        ]

        # Sort by epoch number
        model_files = sorted(
            model_files,
            key=lambda x: int(re.search(r"epoch_(\d+)", x).group(1))
        )

        for file in model_files:

            epoch = int(re.search(r"epoch_(\d+)", file).group(1))

            test_model = copy.deepcopy(self.model)

            checkpoint = torch.load(
                os.path.join(model_dir, file),
                map_location=self.device
            )

            test_model.load_state_dict(checkpoint["model"])

            tester = Tester(
                test_model,
                self.test_loader,
                self.num_classes,
                self.criterion,
                self.device
            )

            result = tester.test(
                return_predictions=return_predictions
            )

            result["epoch"] = epoch
            result["file"] = file

            results.append(result)

            print(f"Finished testing epoch {epoch}")

        return results