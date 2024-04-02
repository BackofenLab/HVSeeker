import pandas as pd

# Specify the paths to the CSV files
sequences_csv_path = 'benchmark_X.csv'
labels_csv_path = 'benchmark_Y.csv'

# Specify the paths for the output text files
output_class1_path = 'Bacteria_benchmark.txt'
output_class2_path = 'Phage_benchmark.txt'

# Read the sequences and labels CSV files into pandas DataFrames
sequences_df = pd.read_csv(sequences_csv_path)
labels_df = pd.read_csv(labels_csv_path)

data_size = int(0.2*len(sequences_df))

sequences_df = sequences_df[0:data_size]
labels_df = labels_df[0:data_size]

# Assuming the sequences DataFrame has an 'id' column and 'sequence' column
# and the labels DataFrame has a 'label' column.
# We also assume that the rows of both DataFrames correspond to one another.

# Combine the two DataFrames into one based on their index
combined_df = pd.concat([sequences_df, labels_df], axis=1)

# Filter the sequences by class label
class1_df = combined_df[combined_df.iloc[:, 1] == 0]
class2_df = combined_df[combined_df.iloc[:, 1] == 1]

def write_sequences_to_file(df, file_path):
    with open(file_path, 'w') as file:
        # Reset index before iterating to ensure a sequential ID from 0 onwards
        df = df.reset_index(drop=True)
        for idx, row in df.iterrows():
            # Write a sequential ID based on the DataFrame's index after reset
            file.write(f">id:{idx}\n")
            # Assuming that the sequence is in the first column (index 0)
            file.write(f"{row.iloc[0]}\n")

# Write the sequences for each class to their respective text files
write_sequences_to_file(class1_df, output_class1_path)
write_sequences_to_file(class2_df, output_class2_path)

print(f"Sequences for class1 have been saved to {output_class1_path}")
print(f"Sequences for class2 have been saved to {output_class2_path}")
