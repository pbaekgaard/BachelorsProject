from components.make_sequences import make_sequences
from sklearn.preprocessing import LabelEncoder
from components.activitydatamodule import ActivityDataModule
from components.activitypredictor import ActivityPredictor
import pytorch_lightning as pl
from pytorch_lightning import loggers

label_encoder = LabelEncoder()
N_EPOCH = 1
BATCH_SIZE = 32


# Fit First
def fitFirst(folder: str):
    global label_encoder
    seq, seq_test, label_encoder = make_sequences(folder, label_encoder)
    data_module = ActivityDataModule(seq, seq_test, BATCH_SIZE)
    model = ActivityPredictor(n_features=6, n_classes=len(label_encoder.classes_))  # type: ignore
    logger = loggers.TensorBoardLogger("lightning_logs", name="activity_classifier")
    trainer = pl.Trainer(max_epochs=N_EPOCH, logger=logger)
    trainer.fit(model, data_module)
    trainer.test(model, data_module.test_dataloader())


# Fit and Predict

fitFirst("ProcessedData")
