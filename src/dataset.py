import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import Dataset

class ProteinDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, bow = False, batch_size=32, max_length=128):
        super(ProteinDataModule, self).__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.max_length = max_length
        self.bow = bow

    def setup(self, stage=None):
        self.train_dataset = ProteinDataset(self.train_df, self.max_length, bow=self.bow)
        self.val_dataset = ProteinDataset(self.val_df, self.max_length, bow=self.bow)
        self.test_dataset = ProteinDataset(self.test_df, self.max_length, bow=self.bow)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class ProteinDataset(Dataset):
    def __init__(self, dataframe, max_length, bow=False):
        self.data = dataframe
        self.max_length = max_length

        if bow:
            self.vectorizer = CountVectorizer(analyzer='char')
            self.vectorizer.fit(self.data['sequence'])
            self.bow = bow
        else:
            self.tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]['sequence']
        label = self.data.iloc[idx]['class_encoded']
        if self.bow:
            inputs = self.vectorizer.transform([sequence]).toarray()
            inputs = torch.tensor(inputs, dtype=torch.float32).squeeze()
        else:
            inputs = self.tokenizer(sequence, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
            inputs = {key: val.squeeze() for key, val in inputs.items()}
        return inputs, torch.tensor(label, dtype=torch.long)
