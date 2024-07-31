import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import numpy as np

def preprocess_data(input_file, output_file):
    # Load the data
    data = pd.read_csv(input_file)

    # Handle missing values
    data = data.dropna()

    # Encode the family_accession to numeric labels
    label_encoder = LabelEncoder()
    data['class_encoded'] = label_encoder.fit_transform(data['family_accession'])

    # Keep only sequences and encoded labels
    data = data.drop(columns=["family_id","sequence_name","family_accession"])

    # Calculate class distribution
    class_counts = data['class_encoded'].value_counts()

    # Apply oversampling to the minority classes with at least 2 examples
    max_count = class_counts.max()
    oversampled_data = pd.DataFrame()

    for class_id in class_counts.index:
        class_data = data[data['class_encoded'] == class_id]
        if len(class_data) < max_count and len(class_data) > 1:
            oversampled_class_data = resample(class_data,
                                              replace=True,  # Sample with replacement
                                              n_samples=20,  # Match median class count
                                              random_state=42)  # Reproducible results
        else:
            oversampled_class_data = class_data
        oversampled_data = pd.concat([oversampled_data, oversampled_class_data])

    # Shuffle the oversampled data
    oversampled_data = oversampled_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the preprocessed data
    oversampled_data.to_csv(output_file, index=False)

    # Save the label encoder mapping
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    with open(output_file.replace('.csv', '_label_mapping.txt'), 'w') as f:
        for key, value in label_mapping.items():
            f.write(f"{key}: {value}\n")

    # Determine class weights
    oversample_class_counts = oversampled_data['class_encoded'].value_counts()
    class_weights = 1. / oversample_class_counts
    class_weights /= class_weights.sum()

    # Save the class weights
    class_weights_dict = {label: weight for label, weight in zip(oversample_class_counts.index, class_weights)}
    with open(output_file.replace('.csv', '_class_weights.txt'), 'w') as f:
        for key, value in class_weights_dict.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess protein data")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output CSV file")
    args = parser.parse_args()

    preprocess_data(args.input_file, args.output_file)
