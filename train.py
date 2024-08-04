from collections import OrderedDict
import warnings

import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pytorch_lightning as pl
from transformers import AutoTokenizer
from lightning.pytorch.loggers import WandbLogger
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.dataset import ProteinDataModule
from src.models.linear_classifier import LinearClassifier
from src.models.transformers_module import TransformersLightningModule
from src.models.ropeformer import TransformerEncoderRoPE


@hydra.main(config_path="config", config_name="config")
def train(cfg: DictConfig):
    """
    Train a model for protein family classification based on the specified model name.

    This function handles the training process for different models, including data loading,
    model initialization, training loop, and evaluation. The available models are:
    - 'baseline': A simple linear classifier.
    - 'esm2': QLoRA fine-tuning of the ESM-2 8M model.

    /!\ To implement :
    - 'ropeformer': A custom implementation of a Transformer with Rotary Position Embeddings.

    Args:
        model_name (str): The name of the model to train. Must be one of 'baseline', 'esm2', or 'ropeformer'.

    Raises:
        ValueError: If an invalid model name is provided.

    Example:
        >>> python train.py --model_name baseline
    """
    # Load and preprocess the data
    train_data = pd.read_csv(to_absolute_path(cfg.dataset.train_path))
    test_data = pd.read_csv(to_absolute_path(cfg.dataset.test_path))
    val_data = pd.read_csv(to_absolute_path(cfg.dataset.val_path))

    # Encode the labels
    label_encoder = LabelEncoder()
    test_data['class_encoded'] = label_encoder.fit_transform(test_data['class_encoded'])
    train_data['class_encoded'] = label_encoder.transform(train_data['class_encoded'])
    val_data['class_encoded'] = label_encoder.transform(val_data['class_encoded'])

    # Define the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')

    # Load class weights
    with open(to_absolute_path(cfg.dataset.class_weights_path), 'r') as f:
        class_weights = {int(line.split(': ')[0]): float(line.split(': ')[1]) for line in f.readlines()}
    class_weights_ordered = OrderedDict(sorted(class_weights.items()))
    weights = torch.tensor([class_weights_ordered[i] for i in range(len(class_weights_ordered))], dtype=torch.float32)

    # Initialize the PyTorch Lightning trainer
    logger = WandbLogger(project="PFAM") if cfg.train.logger == "wandb" else None
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator='cuda',
        gradient_clip_val=cfg.train.gradient_clip_val, # Ensure gradient clipping to avoid exploding gradients
        accumulate_grad_batches=cfg.train.accumulate_grad_batches, # Accumulate grad for larger effective batch size
        logger=logger
    )

    num_classes = len(label_encoder.classes_)
    aa_count = 25

    if cfg.model.name == 'baseline':
        data_module = ProteinDataModule(train_data, val_data, test_data, batch_size=cfg.train.batch_size, bow=True)
        input_size = aa_count
        model = LinearClassifier(input_size, num_classes, weights)   
    elif cfg.model.name == 'esm2':
        data_module = ProteinDataModule(train_data, val_data, test_data, batch_size=cfg.train.batch_size)
        model = TransformersLightningModule(model_name='facebook/esm2_t6_8M_UR50D', num_labels=len(label_encoder.classes_), class_weights=weights)
    else:
        raise ValueError("Model name doesn't match with any available model")

    trainer.fit(
      model=model, 
      datamodule=data_module
    )
    trainer.test(datamodule=data_module)
    


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    train()
