# HVSeeker   
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
Other dependencies as listed in **requirements.yml**    
  
  
  
## Installation  
To set up your environment for this tool, follow these steps:  
  
**Clone the repository:**    

```
git clone https://github.com/bulatef/HVSeeker.git
cd HVSeeker/HVSeeker-DNA
```

**Install required Python packages using pip:**    

```
pip install -r requirements.yml
```
  
**Install required Python using conda:**    


To install required python packages we recommend the use of miniconda


**Creating a Miniconda environment:**


First we install Miniconda for python 3. Miniconda can be downloaded from here:

https://docs.conda.io/en/latest/miniconda.html

Then Miniconda should be installed. On a linux machine the command is similar to this one:
```
bash Miniconda3-latest-Linux-x86_64.sh
```
Then we create an environment. The necessary setup is provided in the "environment.yml" file inside the "for_environment" directory

In order to install the corresponding environment one can execute the following command from the "for_environment" directory

```
conda env create -f environment.yml --name HVSeekerDNA
```

or alternatively:

```
conda env create -f environment.yml --name HVSeekerProt
```

### Activation of the environment

Before running DeepDefense one need to activate the corresponding environment.

```
conda activate HVSeekerDNA
```
or alternatively:

```
conda activate HVSeekerProt
```
  
  
## Usage  
This script can be run from the command line with various options. Below is a detailed description of the command-line arguments:  
  
**Basic Usage HVSeeker-DNA**  
```
python main.py -f [List of training directories] [OPTIONS]
``` 
    
## Options  
| Short Flag | Long Flag       | Description                                                                                   | Default   |
|------------|-----------------|-----------------------------------------------------------------------------------------------|-----------|
| -f         | --train_files   | Specify the directories containing training files. Multiple directories can be specified.     |           |
| -m         | --method        | Choose the preprocessing method (1 = padding, 2 = contigs assembly, 3 = sliding window).      | 1         |
| -l         | --gene_length   | Specify the gene length for training and testing.                                             | 1000      |
| -w         | --window        | Window size for the sliding window method. It adjusts automatically if not valid. Only applicable if method 3 is selected. |     gene_length/10      |
| -vts       | --split         | Specify the validation and test split percentages respectively.                               | [10, 10]  |
| -predict   | --predict_mode  | Use this flag to enable prediction mode using trained models.                                 |           |

When using the -predict flag, the script enters prediction mode. This mode expects the following files in the script directory:
  
**X_test.csv:** A tab-delimited file with two columns; the first column is 'ID' and the second column is 'sequence'.  
**Y_test.csv:** A tab-delimited file with two columns; the first column is 'ID' and the second column is 'class name' (Bacteria or Phage).  
**model_best_acc2_test_model.pt:** A pre-trained model file.  

## Example  
Training a model with default settings on specified files:  

  ```
python main.py -f Bacteria Phage -vts 10 10 -m 1 -l 1000
```
  
Predicting using a pre-trained model:
  
```
python main.py -predict
```


**Basic Usage HVSeeker-Proteins**  

To run HVSeeker-Proteins for further validation of sequences, simply run:


```
python predict.py --output_file {prefix of output file} --test_file {test_file}
```
additionally we provide a file for optimizing and training on a novel dataset:

```
optimize_finetuning.py -o {output_file}
```
and for training:

```
train.py -t {test_file} -f {training_file}
```



