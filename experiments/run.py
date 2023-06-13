import datetime
from pathlib import Path
from typing import Optional

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger

from data_module import TextsDataModule
from model import TransformerClassifier

MLFLOW_URI: Optional[str] = None

# max batch size for our GPU left here, not used in configuration
models_batches = [
    ("mT5", 10),
    ("XLM-R-dist", 12),
    ("XLM-R", 12),
    ("mUSE-dist", 28),
    ("LaBSE", 6),
    ("mBERT", 14),
    ("DistilmBERT", 28),
    ("MPNet", 12),
]

experiment_name = "finetuning_semeval_2017"
experiments = [
    *[
        {
            "name": "Finetuning",
            "dropout": 0.2,
            "freeze_emb": False,
            "type": "Linear",
            "embedding_size": 768,
            "hidden_size": 768,
            "transformer": transformer,
            "batch_size": 6,
            "metric_average": "macro",
            "max_epochs": 3,
            "monitor": "val_f1",
            "seed": 42,
            "deterministic": True,
            "learning_rate": 1e-5,
            "undersampling": 0.01,
        }
        for transformer, _ in models_batches
    ],
    *[
        {
            "name": "JustHead",
            "dropout": 0.2,
            "freeze_emb": True,
            "type": "Linear",
            "embedding_size": 768,
            "hidden_size": 768,
            "transformer": transformer,
            "batch_size": 200,
            "metric_average": "macro",
            "max_epochs": 15,
            "monitor": "val_f1",
            "seed": 42,
            "deterministic": True,
            "learning_rate": 1e-3,
            "undersampling": 0.01,
        }
        for transformer, _ in models_batches
    ],
    *[
        {
            "name": "JustHead",
            "dropout": 0.5,
            "freeze_emb": True,
            "type": "BiLSTM",
            "embedding_size": 768,
            "hidden_size": 32,
            "transformer": transformer,
            "batch_size": 200,
            "metric_average": "macro",
            "max_epochs": 15,
            "monitor": "val_f1",
            "seed": 42,
            "deterministic": True,
            "learning_rate": 5e-3,
            "undersampling": 0.01,
        }
        for transformer, _ in models_batches
    ],
]

for exp in experiments:
    model_name = TransformerClassifier.get_transformer_by_name(exp["transformer"])
    seed_everything(seed=exp["seed"], workers=True)

    run_name = (
        f"{exp['transformer']}"
        + f"_{exp['name']}"
        + f"_{exp['type']}"
        + f'_seed:{exp["seed"]}'
        + f"_{datetime.datetime.now()}"
    )

    data_module = TextsDataModule(
        batch_size=exp["batch_size"],
        tokenizer=model_name,
        num_workers=20,
        undersampling=exp["undersampling"],
    )

    exp["base_transformer"] = model_name

    model = TransformerClassifier.get_model_by_type(exp["type"], exp)
    model.set_ds_encoder(data_module.dataset_encoder)

    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=MLFLOW_URI,
    )
    mlf_logger.log_hyperparams(exp)
    csv_logger = CSVLogger("logs")

    checkpoint_callback = ModelCheckpoint(
        monitor=str(exp["monitor"]),
        dirpath=Path("checkpoints") / experiment_name / run_name,
        filename="{epoch}-{val_f1:.2f}",
        save_top_k=1,
    )
    trainer = Trainer(
        deterministic=exp["deterministic"],
        logger=[csv_logger, mlf_logger],
        max_epochs=exp["max_epochs"],
        callbacks=[checkpoint_callback],
        gpus=-1,
    )
    trainer.fit(model, data_module)
    trainer.test(ckpt_path=checkpoint_callback.best_model_path)
