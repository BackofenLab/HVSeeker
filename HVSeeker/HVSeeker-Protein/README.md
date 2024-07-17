# HVSeeker-proteins   
This tool is designed for training machine learning models on DNA sequence data and making predictions using pre-trained models. The tool supports various preprocessing methods, and allows for model training and prediction.
  
  

  
## Installation  
To set up your environment for this tool, follow these steps:  
  
**Clone the repository:**    

```
git clone https://github.com/bulatef/HVSeeker.git
cd HVSeeker/HVSeeker-DNA
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
conda env create -f environment.yml --name HVSeekerProt
```

### Activation of the environment

Before running DeepDefense one need to activate the corresponding environment.


```
conda activate HVSeekerProt
```
  
  
**Basic Usage HVSeeker-Proteins**  


Since HVSeeker-Proteins relies on ProtBert you will first have to clone the ProtBert github from here: https://github.com/nadavbra/protein_bert

To run HVSeeker-Proteins you will also have to download the pretrained models from: https://drive.google.com/drive/folders/1akwf7QjDA_Hb2VMDhBZEGK7esNWNj3FI?usp=sharing
Then you can simply run the model using the following commands:



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



### Output

The script predict.py creates a table that summarizes the sequence, label and prediction 



