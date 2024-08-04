import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import CountVectorizer

class ProteinDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, bow=False, ropeformer=False, batch_size=32, max_length=128):
        """
        Initialize the Protein Data Module.

        Args:
            train_df (pd.DataFrame): Training dataframe.
            val_df (pd.DataFrame): Validation dataframe.
            test_df (pd.DataFrame): Test dataframe.
            bow (bool): If True, use Bag of Words encoding. Default is False.
            ropeformer (bool): If True, use RoPE embeddings. Default is False.
            batch_size (int): Batch size for the data loaders. 
            max_length (int): Maximum sequence length for tokenization.
        """
        super(ProteinDataModule, self).__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.max_length = max_length
        self.bow = bow
        self.ropeformer = ropeformer

    def setup(self, stage=None):
        """
        Set up the datasets for training, validation, and testing.

        Args:
            stage (str, optional): Stage of the setup (fit, validate, test). Default is None.
        """
        self.train_dataset = ProteinDataset(self.train_df, self.max_length, bow=self.bow, ropeformer=self.ropeformer)
        self.val_dataset = ProteinDataset(self.val_df, self.max_length, bow=self.bow, ropeformer=self.ropeformer)
        self.test_dataset = ProteinDataset(self.test_df, self.max_length, bow=self.bow, ropeformer=self.ropeformer)

    def train_dataloader(self):
        """
        Create the training DataLoader.

        Returns:
            DataLoader: DataLoader for training data.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """
        Create the validation DataLoader.

        Returns:
            DataLoader: DataLoader for validation data.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        """
        Create the test DataLoader.

        Returns:
            DataLoader: DataLoader for test data.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class ProteinDataset(Dataset):
    def __init__(self, dataframe, max_length, bow=False, ropeformer=False):
        """
        Initialize the Protein Dataset.

        Args:
            dataframe (pd.DataFrame): Dataframe containing sequences and labels.
            max_length (int): Maximum sequence length for tokenization.
            bow (bool): If True, use Bag of Words encoding. 
            ropeformer (bool): If True, use RoPE embeddings. 
        """
        self.data = dataframe
        self.max_length = max_length
        self.bow = bow
        self.ropeformer = ropeformer

        if self.bow:
            self.vectorizer = CountVectorizer(analyzer='char')
            self.vectorizer.fit(self.data['sequence'])
        else:
            self.tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample.
        
        Returns:
            dict or torch.Tensor: Tokenized input or Bag of Words vector.
            torch.Tensor: Label corresponding to the input.
        """
        sequence = self.data.iloc[idx]['sequence']
        label = self.data.iloc[idx]['class_encoded']
        
        if self.bow:
            # If using Bag of Words encoding
            inputs = self.vectorizer.transform([sequence]).toarray()
            inputs = torch.tensor(inputs, dtype=torch.float32).squeeze()
        elif not self.bow and not self.ropeformer:
            # If using Huggingface Transformer PT Lightning wrapper
            inputs = self.tokenizer(sequence, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
            inputs = {key: val.squeeze() for key, val in inputs.items()}
        else:
            # If tokenizing without the wrapper for the RoPE transformer
            inputs = self.tokenizer(sequence, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        
        return inputs, torch.tensor(label, dtype=torch.long)

