import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from Bio import SeqIO
import re

# Assuming you have a test fasta file for bacteria and phage
bacteria_test_fasta = 'Bacteria_benchmark.txt'
phage_test_fasta = 'Phage_benchmark.txt'
# Configuration
NUC_ORDER = {y: x for x, y in enumerate(["A", "T", "C", "G"])}
NUC_COUNT = len(NUC_ORDER)

# Load the trained model
model = load_model('seeker_model.h5')

def seq2matrix(sequence, nuc_order, fragment_length):
    assert len(sequence) <= fragment_length
    sequence_len = min(len(sequence), fragment_length)
    sequence = sequence[:sequence_len]
    ret = np.zeros((4, fragment_length))
    for idx, base in enumerate(sequence):
        ret[nuc_order[base], idx] = 1
    return ret

# Function to prepare the test data
def prepare_test_data(fasta_path, nuc_order, fragment_length):
    entries = list(SeqIO.parse(fasta_path, "fasta"))
    matrix_data = [seq2matrix(str(entry.seq).upper(), nuc_order, fragment_length) for entry in entries if re.match('^[ACGT]+$', str(entry.seq))]
    return np.array(matrix_data)

# Prepare test data
bacteria_test_data = prepare_test_data(bacteria_test_fasta, NUC_ORDER, 1000)
phage_test_data = prepare_test_data(phage_test_fasta, NUC_ORDER, 1000)

# Combine test data and create labels
X_test = np.concatenate((phage_test_data, bacteria_test_data))
y_test = np.concatenate((np.ones(len(phage_test_data)), np.zeros(len(bacteria_test_data))))

print(X_test[0:5])
# Make predictions
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).astype(int)  # Convert predictions to binary labels

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_reduced = conf_matrix[1:, 1:]


# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_reduced, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Save the confusion matrix figure before showing it
plt.savefig('confusion_matrix_benchmark.png')

# Display the plot
plt.show()
