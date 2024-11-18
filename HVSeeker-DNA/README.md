# HVSeeker-DNA   
This tool is designed for training machine learning models on DNA sequence data and making predictions using pre-trained models. The tool supports various preprocessing methods, and allows for model training and prediction.
  
## Features  
**Multiple Preprocessing Methods:** Choose from padding, contigs assembly, or sliding window approaches.  
**Model Training and Prediction:** Train models from scratch or use existing models to make predictions.  
**Customizable Sequence Lengths and Window Sizes:** Adjust based on your dataset needs.  
  
  
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

## Run HVSeekerDNA with docker

Before trying to create the image make sure you have both docker and nvidia-container-toolkit installed:

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
sudo docker run  --gpus all -v {your-system-path}/Sample_Data/Bacteria:/app/Bacteria -v {your-system-path}/Sample_Data/Phage:/app/Phage -v {your-system-path}:/app/output hvseekerdna   python -u main.py -f Bacteria Phage -ts 10 10 -m 1 -l 1000 -o output
```

and for prediction:

```
sudo docker run --gpus all -v {your-system-path}Sample_Data:/app/Sample_Data -v {your-system-path}:/app/output hvseekerdna python -u main.py -predict -o output
```

we made the build docker image available here: https://drive.google.com/file/d/1OPPmD_s9YYOKd4leQGk4UkLFPSrEUS7O/view?usp=sharing


## Output  
The script outputs the trained model parameters, performance metrics, and predictions (if in prediction mode). All outputs are stored in the project directory.
