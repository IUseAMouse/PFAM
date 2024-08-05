import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer


def rotary_position_embeddings(dim, seq_len):
    """
    Generates rotary position embeddings.

    Args:
        dim (int): The dimensionality of the embeddings.
        seq_len (int): The length of the sequence.

    Returns:
        torch.Tensor: The rotary position embeddings with shape (1, seq_len, dim).
    """
    # Compute the inverse frequency for each dimension
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

    # Create a tensor of shape (seq_len) containing the positions [0, 1, 2, ..., seq_len-1]
    t = torch.arange(seq_len, dtype=torch.float32)

    # Compute the outer product of positions and inverse frequencies
    freqs = torch.einsum('i,j->ij', t, inv_freq)

    # Concatenate frequencies with themselves along the last dimension to match the model dimension
    emb = torch.cat((freqs, freqs), dim=-1)

    # Add batch and sequence dimensions
    return emb.unsqueeze(0)


def apply_rotary_pos_emb(x, sincos):
    """
    Applies rotary position embeddings to input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
        sincos (tuple): Tuple containing sin and cos tensors.

    Returns:
        torch.Tensor: Tensor with applied rotary position embeddings.
    """
    sin, cos = sincos

    # Apply the rotary position embeddings to the input tensor
    return (x * cos) + (rotate_half(x) * sin)


def rotate_half(x):
    """
    Rotates half of the dimensions of the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).

    Returns:
        torch.Tensor: Tensor with rotated half dimensions.
    """
    # Split the tensor into two halves along the last dimension
    x1, x2 = x.chunk(2, dim=-1)

    # Concatenate the rotated halves (second half negated and first half unchanged)
    return torch.cat((-x2, x1), dim=-1)


# Custom Transformer Encoder Model with RoPE in PyTorch Lightning
class TransformerEncoderRoPE(pl.LightningModule):
    def __init__(self, input_size, num_classes, class_weights, max_len=256, embedding_dim=128, learning_rate=1e-3):
        """
        Initialize the Transformer Encoder with RoPE (Rotary Position Embeddings).

        Args:
            input_size (int): Size of the input vocabulary.
            num_classes (int): Number of output classes.
            class_weights (torch.Tensor): Class weights for handling class imbalance.
            max_len (int, optional): Maximum sequence length. Defaults to 256.
            embedding_dim (int, optional): Dimension of the embeddings. Defaults to 128.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        """
        super(TransformerEncoderRoPE, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.max_len = max_len
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss(class_weights)
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # Embedding layer
        x = self.embedding(x)
        batch_size, seq_len, dim = x.size()

        # Generate and apply rotary position embeddings
        rotary_emb = rotary_position_embeddings(self.embedding_dim, seq_len).to(x.device)  # Shape: [1, seq_len, dim]
        rotary_emb = rotary_emb.repeat(batch_size, 1, 1)  # Repeat for batch size: [batch_size, seq_len, dim]
        sincos = rotary_emb.sin(), rotary_emb.cos()
        sin, cos = sincos
        x = apply_rotary_pos_emb(x, (sin, cos))

        # Pass through Transformer encoder
        x = self.transformer_encoder(x)

        # Use the representation of the CLS token (first token) for classification
        x = x[:, 0, :]  # Shape: [batch_size, embedding_dim]
        x = self.fc(x)  # Shape: [batch_size, num_classes]
        return x

    def training_step(self, batch, batch_idx):
        """
        Training step for one batch.

        Args:
            batch (tuple): A tuple containing input_ids and labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss.
        """
        input_ids, labels = batch
        outputs = self(input_ids)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for one batch.

        Args:
            batch (tuple): A tuple containing input_ids and labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Validation loss.
        """
        input_ids, labels = batch
        outputs = self(input_ids)
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

    def test_step(self, batch, batch_idx):
        """
        Test step for one batch.

        Args:
            batch (tuple): A tuple containing sequences and labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Test loss.
        """
        sequences, labels = batch
        outputs = self(sequences)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
        precision = precision_score(labels.cpu(), preds.cpu(), average='weighted')
        recall = recall_score(labels.cpu(), preds.cpu(), average='weighted')
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        self.log('test_f1', f1, prog_bar=True)
        self.log('test_precision', precision, prog_bar=True)
        self.log('test_recall', recall, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for the training process.

        Returns:
            torch.optim.Optimizer: Optimizer used for training.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

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
