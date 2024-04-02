import csv

input_csv_path = 'Y_test.csv'  # Replace with your input CSV file path
output_csv_path = 'Y_test_corrected.csv'  # Replace with your desired output CSV file path

# Define the mapping from original labels to numeric labels
label_mapping = {
    'Bacteria': 0,
    'Phage': 1
}

# Function to detect delimiter
def detect_delimiter(file_path):
    with open(file_path, 'r') as file:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(file.readline(), delimiters=',\t')
        return dialect.delimiter

# Detect the delimiter of the input CSV file
delimiter = detect_delimiter(input_csv_path)

# Read the input CSV file and write the mapped values to the output CSV file
with open(input_csv_path, mode='r', newline='') as infile, \
     open(output_csv_path, mode='w', newline='') as outfile:

    # Create CSV reader with detected delimiter and writer
    reader = csv.reader(infile, delimiter=delimiter)
    writer = csv.writer(outfile)

    # Read each row in the CSV, map the values, and write to the output file
    for row in reader:
        if row:  # Check if row is not empty
            original_label = row[1]  # Assuming the label is in the second column
            mapped_value = label_mapping.get(original_label, 'Unknown')  # Default to 'Unknown' if label is not in mapping
            # Write only the mapped value
            writer.writerow([mapped_value])

print(f"Mapping complete. Mapped values have been saved to {output_csv_path}")
