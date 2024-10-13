# HVSeeker   
**Motivation**
Bacteriophages are among the most abundant organisms on Earth, significantly impacting
ecosystems and human society. The identification of viral sequences, especially novel ones, from mixed
metagenomes is a critical first step in analyzing the viral components of host samples. This plays a
key role in many downstream tasks. However, this is a challenging task due to their rapid evolution
rate. The identification process typically involves two steps: distinguishing viral sequences from the host
and identifying if they come from novel viral genomes. Traditional metagenomic techniques that rely
on sequence similarity with known entities often fall short, especially when dealing with short or novel
genomes. Meanwhile, deep learning has demonstrated its efficacy across various domains, including the
Bioinformatics field
**Results:**
We have developed HVSeeker, a deep learning-based method for distinguishing between
bacterial and phage sequences. HVSeeker consists of two separate models: one analyzing DNA
sequences and the other focusing on proteins. This method has shown promising results on sequences
with various lengths, ranging from 200 to 1500 base pairs. Tested on both NCBI and IMGVR
databases, HVSeeker outperformed several methods from the literature such as Seeker, Rnn-VirSeeker,
DeepVirFinder, and PPR-Meta. Moreover, when compared with other methods on benchmark datasets,
HVSeeker has shown better performance, establishing its effectiveness in identifying unknown phage
genomes

## Features  
**Multiple Preprocessing Methods:** Choose from padding, contigs assembly, or sliding window approaches.  
**Model Training and Prediction:** Train models from scratch or use existing models to make predictions.  
**Customizable Sequence Lengths and Window Sizes:** Adjust based on your dataset needs.  

## Hardware
We are currently using one single NVIDIA Tesla T4, 16 GB GPU for training. If you only want to predict and test using HVSeeker you can also make use of your CPU.
  
## Prerequisites
Before you run this script, ensure you have the following installed:  
  
**Python 3.x**  
**PyTorch**  
**BioPython**  
Other dependencies for HVSeeker-DNA are listed in **HVSeeker-DNA_environment.yml** and for HVSeeker-Protein in **HVSeeker_Prot_enviroment.yml**

  
  
  
## Installation  
To set up your environment for this tool, follow these steps:  
  
**Clone the repository:**    

```
git clone https://github.com/bulatef/HVSeeker.git
cd HVSeeker/HVSeeker-DNA
```

**Install required Python packages using pip:**    

```
pip install -r HVSeeker-DNA_environment.yml
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
conda env create -f HVSeeker-DNA_environment.yml --name HVSeekerDNA
```

or alternatively:

```
conda env create -f HVSeeker_Prot_enviroment.yml --name HVSeekerProt
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

## Run HVSeekerDNA with docker

Alternatively we provide a docker container capable of running HVSeeker_Protein. You can build it using the Dockerfile provided in the docker folder.
First move the Dockerfile and enviroment.yml into the HVSeeker_Protein main directory. Before trying to create the image make sure you have both docker and nvidia-container-toolkit installed:

```

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```


Alternative to using conda we make a docker image available. You can either build the docker image directly after copying docker file and environment into the corresponding folder:

```
docker build --network host hvseekerdna .
```
and then run all scripts using the created image. Just make sure to set the volume accordingly to your system. For training:

```
sudo docker run  --gpus all -v {your-system-path}/Sample_Data/Bacteria:/app/Bacteria -v {your-system-path}/Sample_Data/Phage:/app/Phage -v {your-system-path}:/app/output -v {your-system-path}:/app/data_path hvseekerdna   python -u main.py -f Bacteria Phage -ts 10 10 -m 1 -l 1000 -o output -d data_path
```

and for prediction:

```
sudo docker run --gpus all -v {your-system-path}Sample_Data:/app/Sample_Data -v {your-system-path}:/app/output -v {your-system-path}:/app/data_path hvseekerdna python -u main.py -predict -d data_path -o output
```

## Run HVSeekerProt with docker

Alternatively we provide a docker container capable of running HVSeeker_Protein. You can build it using the Dockerfile provided in the docker folder.
First move the Dockerfile and enviroment.yml into the HVSeeker_Protein main directory. Make sure you have docker and and nvidia-container-toolkit installed:

```
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

then build the docker image:

```
sudo docker build --network host -t hvseekerprot .

```
and run with for training:

```
sudo docker run --network host --gpus all --privileged=true -v {your-path}:/app/input -v {your-path-to-models}:/app/models hvseekerprot  bash -c "yes 'Yes' |  python -u train.py -t input/{data} -f input/{data} -s models"
```

or prediction:

```
sudo docker run --network host --gpus all -v {your-path}:/app/input -v {your-path-to-models}:/app/models hvseekerprot  bash -c "yes 'Yes' |  python -u predict.py --test_file input/{test_file} --output_file input/{output name} -m models/{model_name}
```

We made the build image available here:

  
  
## Usage  
This script can be run from the command line with various options. Below is a detailed description of the command-line arguments:  
  
**Basic Usage HVSeeker-DNA**  

To use HVSeeker you can either train models yourself or download our pretrained models from https://drive.google.com/drive/folders/14qXKk4YnBaxZ0_OPxDe-frm0WpQNnP6w in the folder /Method/Proposed method/


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


## Example  
Training a model with default settings on specified files:  

```
python main.py -f Bacteria Phage -vts 10 10 -m 1 -l 1000
```

Predicting using a pre-trained model:
  
```
python main.py -predict
```






