import csv

# Function to encode DNA sequence
def encode_dna_sequence(seq):
    encoding = {'A': '1', 'C': '2', 'G': '3', 'T': '4'}
    return [encoding.get(nuc, '0') for nuc in seq]  # Return a list of encoded digits

input_file_path = 'benchmark_X.csv'
output_file_path = 'encoded_x_test_benchmark.csv'

# Open the input file for reading
with open(input_file_path, mode='r', newline='') as infile:

    # Set up CSV reader with comma delimiter (update as needed)
    reader = csv.reader(infile, delimiter='\t')
    
    with open(output_file_path, mode='w', newline='') as outfile:
        # Set up CSV writer
        writer = csv.writer(outfile)
        
        # Process each DNA sequence in the file
        for row in reader:
            if row:  # checking if the row is not empty
                dna_sequence = row[0]  # read the second column
                encoded_sequence = encode_dna_sequence(dna_sequence)
                # Write the encoded sequence as individual integers
                writer.writerow(encoded_sequence)

print("Encoding complete. Encoded sequences have been saved to", output_file_path)
