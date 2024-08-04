import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class TransformersLightningModule(pl.LightningModule):
    def __init__(self, model_name, num_labels, class_weights):
        """
        Initialize the Transformers Lightning Module with LoRA and HuggingFace.

        Args:
            model_name (str): The name of the pretrained model.
            num_labels (int): The number of output labels.
            class_weights (torch.Tensor): Class weights for handling class imbalance.
        """
        super(TransformersLightningModule, self).__init__()
        
        # Configuration for Low-Rank Adaptation (LoRA)
        self.lora_config = lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["query", "key", "value"],
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_CLS"  
        )
        
        # Load pretrained model for sequence classification
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
  
        # Prepare the model for quantized training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, lora_config)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set learning rate
        self.learning_rate = 0.001

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input tensor containing token IDs.
            attention_mask (torch.Tensor): Attention mask tensor.
            labels (torch.Tensor, optional): Labels tensor. Defaults to None.
        
        Returns:
            transformers.modeling_outputs.SequenceClassifierOutput: Model outputs including loss and logits.
        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        """
        Training step for one batch.

        Args:
            batch (tuple): A tuple containing input_ids, attention_mask, and labels.
            batch_idx (int): Index of the batch.
        
        Returns:
            torch.Tensor: Training loss.
        """
        inputs, labels = batch
        outputs = self(inputs['input_ids'], inputs['attention_mask'], labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for one batch.

        Args:
            batch (tuple): A tuple containing input_ids, attention_mask, and labels.
            batch_idx (int): Index of the batch.
        
        Returns:
            torch.Tensor: Validation loss.
        """
        inputs, labels = batch
        outputs = self(inputs['input_ids'], inputs['attention_mask'], labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
        precision = precision_score(labels.cpu(), preds.cpu(), average='weighted')
        recall = recall_score(labels.cpu(), preds.cpu(), average='weighted')
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_epoch=True, prog_bar=True)
        self.log('val_precision', precision, on_epoch=True, prog_bar=True)
        self.log('val_recall', recall, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step for one batch.

        Args:
            batch (tuple): A tuple containing input_ids, attention_mask, and labels.
            batch_idx (int): Index of the batch.
        
        Returns:
            torch.Tensor: Test loss.
        """
        inputs, labels = batch
        outputs = self(inputs['input_ids'], inputs['attention_mask'], labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
        precision = precision_score(labels.cpu(), preds.cpu(), average='weighted')
        recall = recall_score(labels.cpu(), preds.cpu(), average='weighted')
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_epoch=True, prog_bar=True)
        self.log('test_f1', f1, on_epoch=True, prog_bar=True)
        self.log('test_precision', precision, on_epoch=True, prog_bar=True)
        self.log('test_recall', recall, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for the training process.
        
        Returns:
            torch.optim.Optimizer: Optimizer used for training.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def on_after_backward(self):
        """
        Hook to clip gradients after backpropagation and log gradient norms.
        """
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Log the gradient norms
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.log(f'grad_norm_{name}', param.grad.norm(2).item())
