# HVSeeker-DNA   
This tool is designed for training machine learning models on DNA sequence data and making predictions using pre-trained models. The tool supports various preprocessing methods, and allows for model training and prediction.
  
## Features  
**Multiple Preprocessing Methods:** Choose from padding, contigs assembly, or sliding window approaches.  
**Model Training and Prediction:** Train models from scratch or use existing models to make predictions.  
**Customizable Sequence Lengths and Window Sizes:** Adjust based on your dataset needs.  
  
## Prerequisites
Before you run this script, ensure you have the following installed:  
  
**Python 3.x**  
**PyTorch**  
**BioPython**  
Other dependencies as listed in **requirements.txt**    
  
## Installation  
To set up your environment for this tool, follow these steps:  
  
**Clone the repository:**    

```
git clone https://github.com/bulatef/HVSeeker.git
cd HVSeeker/HVSeeker-DNA
```

**Install required Python packages:**    

```
pip install -r requirements.txt
```
  
## Usage  
This script can be run from the command line with various options. Below is a detailed description of the command-line arguments:  
  
**Basic Usage**  
```
python main.py -f [List of training directories] [OPTIONS]
``` 
    
## Options  
| Short Flag | Long Flag       | Description                                                                                   | Default   |
|------------|-----------------|-----------------------------------------------------------------------------------------------|-----------|
| -f         | --train_files   | Specify the directories containing training files. Multiple directories can be specified.     |           |
| -m         | --method        | Choose the preprocessing method (1 = padding, 2 = contigs assembly, 3 = sliding window).      | 1         |
| -l         | --gene_length   | Specify the gene length for training and testing.                                             | 1000      |
| -w         | --window        | Window size for the sliding window method. It adjusts automatically if not valid. Only applicable if method 3 is selected. |           |
| -vts       | --split         | Specify the validation and test split percentages respectively.                               | [10, 10]  |
| -predict   | --predict_mode  | Use this flag to enable prediction mode using trained models.                                 |           |

When using the -predict flag, the script enters prediction mode. This mode expects the following files in the script directory:
  
**X_test.csv:** A tab-delimited file with two columns; the first column is 'ID' and the second column is 'sequence'.  
**Y_test.csv:** A tab-delimited file with two columns; the first column is 'ID' and the second column is 'class name' (Bacteria or Phage).  
**model_best_acc2_test_model.pt:** A pre-trained model file.  

## Example  
Training a model with default settings on specified files:  

  ```
python main.py -f "data/dna_sequences" --method 1
Predicting using a pre-trained model:
```
  
Prediction of sequences using a pretrained model:  
  
```
python main.py -predict
```
## Output  
The script outputs the trained model parameters, performance metrics, and predictions (if in prediction mode). All outputs are stored in the project directory.
