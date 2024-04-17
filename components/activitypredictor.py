import pytorch_lightning as pl
from torch import nn
from torchmetrics.functional import accuracy
from components.sequencemodel import SequenceModel
from torch import optim


class ActivityPredictor(pl.LightningModule):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.model = SequenceModel(n_features, n_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.n_classes = n_classes

    def forward(self, x, labels):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        sequences, labels = batch["sequence"], batch["label"]
        loss, outputs = self(sequences, labels)
        predictions = outputs.argmax(dim=1)
        step_accuracy = accuracy(predictions, labels, task="multiclass", num_classes=self.n_classes)

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_accuracy", step_accuracy, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy": step_accuracy}

    def validation_step(self, batch, batch_idx):
        sequences, labels = batch["sequence"], batch["label"]
        loss, outputs = self(sequences, labels)
        predictions = outputs.argmax(dim=1)
        step_accuracy = accuracy(predictions, labels, task="multiclass", num_classes=self.n_classes)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_accuracy", step_accuracy, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy": step_accuracy}

    def test_step(self, batch, batch_idx):
        sequences, labels = batch["sequence"], batch["label"]
        loss, outputs = self(sequences, labels)
        predictions = outputs.argmax(dim=1)
        step_accuracy = accuracy(predictions, labels, task="multiclass", num_classes=self.n_classes)

        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_accuracy", step_accuracy, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy": step_accuracy}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
