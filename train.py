from collections import OrderedDict
import warnings

import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pytorch_lightning as pl
from transformers import AutoTokenizer
from lightning.pytorch.loggers import WandbLogger

from src.dataset import ProteinDataModule
from src.models.linear_classifier import LinearClassifier
from src.models.transformers_module import TransformersLightningModule
from src.models.ropeformer import TransformerEncoderRoPE


def train(model_name: str):
    """
    Train a model for protein family classification based on the specified model name.

    This function handles the training process for different models, including data loading,
    model initialization, training loop, and evaluation. The available models are:
    - 'baseline': A simple linear classifier.
    - 'esm2': QLoRA fine-tuning of the ESM-2 8M model.
    - 'ropeformer': A custom implementation of a Transformer with Rotary Position Embeddings (RoPE).

    Args:
        model_name (str): The name of the model to train. Must be one of 'baseline', 'esm2', or 'ropeformer'.

    Raises:
        ValueError: If an invalid model name is provided.

    Example:
        >>> train('baseline')
        This will train the baseline linear classifier model.
    """
    # Load and preprocess the data
    train_data = pd.read_csv('/content/drive/MyDrive/PFAM - Copie/data/preprocessed/train.csv')
    test_data = pd.read_csv('/content/drive/MyDrive/PFAM - Copie/data/preprocessed/test.csv')
    val_data = pd.read_csv('/content/drive/MyDrive/PFAM - Copie/data/preprocessed/dev.csv')

    # Encode the labels
    label_encoder = LabelEncoder()
    test_data['class_encoded'] = label_encoder.fit_transform(test_data['class_encoded'])
    train_data['class_encoded'] = label_encoder.transform(train_data['class_encoded'])
    val_data['class_encoded'] = label_encoder.transform(val_data['class_encoded'])

    # Define the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')

    # Load class weights
    with open('data/preprocessed/class_weights.txt', 'r') as f:
        class_weights = {int(line.split(': ')[0]): float(line.split(': ')[1]) for line in f.readlines()}
    class_weights_ordered = OrderedDict(sorted(class_weights.items()))
    weights = torch.tensor([class_weights_ordered[i] for i in range(len(class_weights_ordered))], dtype=torch.float32)

    # Initialize the PyTorch Lightning trainer
    logger = WandbLogger(project="PFAM")
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator='cuda',
        gradient_clip_val=1.0, # Ensure gradient clipping to avoid exploding gradients
        accumulate_grad_batches=3, # Accumulate grad for larger effective batch size
        logger=logger
    )

    num_classes = len(label_encoder.classes_)
    aa_count = 25

    if model_name == 'baseline':
        data_module = ProteinDataModule(train_data, val_data, test_data, batch_size=500, bow=True)
        input_size = aa_count
        model = LinearClassifier(input_size, num_classes, weights)   
    elif model_name == 'esm2':
        data_module = ProteinDataModule(train_data, val_data, test_data, batch_size=6, bow=False)
        model = TransformersLightningModule(model_name='facebook/esm2_t6_8M_UR50D', num_labels=len(label_encoder.classes_), class_weights=weights)
    elif model_name == 'ropeformer':
        data_module = ProteinDataModule(train_data, val_data, test_data, batch_size=50, bow=False)
        model = TransformerEncoderRoPE(input_size=256, num_classes=num_classes, class_weights=weights)
    else:
        raise ValueError("Model name doesn't match with any available model")

    trainer.fit(
      model=model, 
      datamodule=data_module
    )
    trainer.test(datamodule=data_module)
    


if __name__ == "__main__":
    import argparse
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description="Train Protein Classifier")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the target model")
    args = parser.parse_args()

    train(args.model_name)
