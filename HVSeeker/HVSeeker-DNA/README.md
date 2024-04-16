HVSeeker-DNA
This Python script (dna_analysis.py) is designed for training machine learning models on DNA sequence data and making predictions using pre-trained models. The script supports various preprocessing methods, handles both DNA and protein sequences, and allows for model training and prediction.

Features
Multiple Preprocessing Methods: Choose from padding, contigs assembly, or sliding window approaches.
Flexible Input Handling: Supports both DNA and protein sequence data.
Model Training and Prediction: Train models from scratch or use existing models to make predictions.
Customizable Sequence Lengths and Window Sizes: Adjust based on your dataset needs.
Prerequisites
Before you run this script, ensure you have the following installed:

Python 3.x
PyTorch
BioPython
Other dependencies as listed in requirements.txt (if available)
Installation
To set up your environment for this tool, follow these steps:

Clone the repository:

bash
Copy
git clone https://github.com/your-repository/dna-sequence-analysis.git
cd dna-sequence-analysis
Install required Python packages:

bash
Copy
pip install -r requirements.txt
Usage
This script can be run from the command line with various options. Below is a detailed description of the command-line arguments:

Basic Usage
bash
Copy
python dna_analysis.py --train_files [List of training directories] [OPTIONS]
Options
-f, --train_files: Specify the directories containing training files. Multiple directories can be specified.
-m, --method: Choose the preprocessing method (1 = padding, 2 = contigs assembly, 3 = sliding window). Default is 1.
-l, --gene_length: Specify the gene length for training and testing. Default is 1000.
-w, --window: Window size for the sliding window method. It adjusts automatically if not valid. Only applicable if method 3 is selected.
-vts, --split: Specify the validation and test split percentages respectively. Default is [10, 10].
--predict_mode: Use this flag to enable prediction mode using trained models.
-p, --protein_sequences: Set this flag if providing protein sequences only.
Prediction Mode
When using the -predict flag, the script enters prediction mode. This mode expects the following files in the script directory:

X_test.csv: A tab-delimited file with two columns; the first column is 'ID' and the second column is 'sequence'.
Y_test.csv: A tab-delimited file with two columns; the first column is 'ID' and the second column is 'class name' (Bacteria or Phage).
model_best_acc2_test_model.pt: A pre-trained model file.
Example
Training a model with default settings on specified files:

bash
Copy
python dna_analysis.py --train_files "data/dna_sequences" --method 1
Predicting using a pre-trained model:

bash
Copy
python dna_analysis.py --predict_mode
Output
The script outputs the trained model parameters, performance metrics, and predictions (if in prediction mode). All outputs are stored in specified or default locations within the project directory.
