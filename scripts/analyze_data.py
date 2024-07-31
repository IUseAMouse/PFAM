import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_data(input_file):
    # Load the data
    data = pd.read_csv(input_file)

    # Display basic information
    print("Basic Information")
    print(data.info())
    print("\n")

    # Display basic statistics
    print("Basic Statistics")
    print(data.describe())
    print("\n")

    # Check for missing values
    print("Missing Values")
    print(data.isnull().sum())
    print("\n")

    # Analyze class distribution based on family_accession
    class_counts = data['family_accession'].value_counts()
    print("Class Distribution")
    print(class_counts)
    print("\n")

    # Display the top 10 most frequent classes
    print("Top 10 Most Frequent Classes")
    print(class_counts.head(10))
    print("\n")

    # Display the bottom 10 least frequent classes
    print("Bottom 10 Least Frequent Classes")
    print(class_counts.tail(10))
    print("\n")

    # Display the median class size
    print("Median class size")
    print(class_counts.median())
    print("\n")

    # Analyze sequence lengths
    data['sequence_length'] = data['sequence'].apply(len)
    max_length = data['sequence_length'].max()
    print(f"Maximum Sequence Length: {max_length}\n")

    # Basic statistics on sequence length
    length_stats = data['sequence_length'].describe()
    print("Sequence Length Statistics")
    print(length_stats)
    print("\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze protein data")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV file")
    args = parser.parse_args()

    analyze_data(args.input_file)
