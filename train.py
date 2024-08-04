import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.dataset import ProteinDataModule
from src.models.linear_classifier import LinearClassifier
from src.models.transformers_module import TransformersLightningModule


def train(baseline=False, custom=False, fine_tune=True):
    # Load and preprocess the data
    train_data = pd.read_csv('/content/drive/MyDrive/PFAM - Copie/data/preprocessed/train.csv')
    test_data = pd.read_csv('/content/drive/MyDrive/PFAM - Copie/data/preprocessed/test.csv')
    val_data = pd.read_csv('/content/drive/MyDrive/PFAM - Copie/data/preprocessed/dev.csv')

    # Encode the labels
    label_encoder = LabelEncoder()
    train_data['class_encoded'] = label_encoder.fit_transform(train_data['class_encoded'])
    test_data['class_encoded'] = label_encoder.transform(test_data['class_encoded'])
    val_data['class_encoded'] = label_encoder.transform(val_data['class_encoded'])

    # Define the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')

    # Load class weights
    with open('/content/drive/MyDrive/PFAM - Copie/data/preprocessed/class_weights.txt', 'r') as f:
        class_weights = {int(line.split(': ')[0]): float(line.split(': ')[1]) for line in f.readlines()}
    class_weights_ordered = OrderedDict(sorted(class_weights.items()))
    weights = torch.tensor([class_weights_ordered[i] for i in range(len(class_weights_ordered))], dtype=torch.float32)

    # Initialize the PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator='auto',
        gradient_clip_val=1.0 # Ensure gradient clipping to avoid exploding gradients
    )

    if baseline:
        model = LinearClassifier(input_size, num_classes, weights)
        data_module = ProteinDataModule(train_data, val_data, test_data, tokenizer, bow=True)
        

    if fine_tune:
        data_module = ProteinDataModule(train_data, val_data, test_data, tokenizer, bow=False)
        model = TransformersLightningModule(model_name='facebook/esm2_t6_8M_UR50D', num_labels=len(label_encoder.classes_), class_weights=weights)

    if custom:
        data_module = ProteinDataModule(train_data, val_data, test_data, tokenizer, bow=False)
        model = LinearClassifier(input_size, num_classes, weights)

        
    trainer.fit(model, data_module)
    trainer.test(datamodule=data_module)
    # Test the model
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess protein data")
    parser.add_argument("--data_file", type=str, required=True, help="Path to train CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the preprocessed files")
    args = parser.parse_args()

    preprocess_data(args.data_file, args.output_dir)
