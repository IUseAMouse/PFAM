import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# RoPE implementation
def rotary_position_embeddings(dim, seq_len):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.einsum('i,j->ij', t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.unsqueeze(0)  # Add batch and sequence dimensions

def apply_rotary_pos_emb(x, sincos):
    sin, cos = sincos
    return (x * cos) + (rotate_half(x) * sin)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

# Custom Transformer Encoder Model with RoPE in PyTorch Lightning
class TransformerEncoderRoPE(pl.LightningModule):
    def __init__(self, input_size, num_classes, max_len=256, embedding_dim=128, learning_rate=1e-3):
        super(TransformerEncoderRoPE, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.max_len = max_len
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.embedding(x)
        batch_size, seq_len, dim = x.size()
        rotary_emb = rotary_position_embeddings(self.embedding_dim, seq_len).to(x.device)  # Shape: [1, seq_len, dim]
        rotary_emb = rotary_emb.repeat(batch_size, 1, 1)  # Repeat for batch size: [batch_size, seq_len, dim]

        sincos = rotary_emb.sin(), rotary_emb.cos()
        sin, cos = sincos
        x = apply_rotary_pos_emb(x, (sin, cos))
        x = self.transformer_encoder(x)
        # Use the representation of the CLS token (first token) for classification
        x = x[:, 0, :]  
        x = self.fc(x)  # Shape: [batch_size, num_classes]
        return x

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        outputs = self(input_ids)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        outputs = self(input_ids)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss)
        return {'loss': loss, 'preds': outputs, 'labels': labels}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])
        preds = torch.argmax(preds, dim=1)
        accuracy = accuracy_score(labels.cpu(), preds.cpu())
        f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
        precision = precision_score(labels.cpu(), preds.cpu(), average='weighted')
        recall = recall_score(labels.cpu(), preds.cpu(), average='weighted')
        self.log('val_accuracy', accuracy)
        self.log('val_f1', f1)
        self.log('val_precision', precision)
        self.log('val_recall', recall)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

# Define the ProteinDataset class
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def preprocess_sequence(self, sequence):
        # Replace special amino acids with 'X'd to fit ProtBert tokenizer requirements
        sequence = sequence.replace('U', 'X').replace('Z', 'X').replace('O', 'X').replace('B', 'X')
        # Add spaces between each amino acid to fit ProtBert tokenizer requirements
        spaced_sequence = ' '.join(sequence)
        return spaced_sequence

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        sequence = self.preprocess_sequence(sequence)
        encoded_input = self.tokenizer(sequence,
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_length,
                                       return_tensors='pt')
        input_ids = encoded_input['input_ids'].squeeze()
        attention_mask = encoded_input['attention_mask'].squeeze()
        return input_ids, label
