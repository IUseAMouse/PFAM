import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

class LinearClassifier(pl.LightningModule):
    def __init__(self, input_size, num_classes, class_weights):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.criterion = nn.CrossEntropyLoss(class_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        sequences, labels = batch
        outputs = self(sequences)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        sequences, labels = batch
        outputs = self(sequences)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
        precision = precision_score(labels.cpu(), preds.cpu(), average='weighted')
        recall = recall_score(labels.cpu(), preds.cpu(), average='weighted')
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        self.log('val_precision', precision, prog_bar=True)
        self.log('val_recall', recall, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences, labels = batch
        outputs = self(sequences)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
        precision = precision_score(labels.cpu(), preds.cpu(), average='weighted')
        recall = recall_score(labels.cpu(), preds.cpu(), average='weighted')
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        self.log('val_precision', precision, prog_bar=True)
        self.log('val_recall', recall, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def on_after_backward(self):
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        # Log the gradient norms
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.log(f'grad_norm_{name}', param.grad.norm(2).item())