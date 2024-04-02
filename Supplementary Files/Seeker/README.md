This directory contains the script to train and test the Seeker model, along with scripts to prepare the dataset for training. To run the script and train a model, use the following command:
```
python train_model.py --bacteria sample_bacteria_training.txt --phage sample_phage_training.txt --out sample_model.h5

```
The model was trained on 452,608 sequences of length 1000, which is the same training data used to train HVSeeker. Additionally, the training model is composed of 5 LSTM units, followed by a dense layer with a tanh activation function.  
The training and testing process undergoes the following steps:  
1. run the dataset_creator.py file. This will prepare the dataset in the same format that is accepted by Seeker.
2. run the train_model.py file for training. The training will go through 100 epochs with a batch size of 27.
3. save the model in the same directory.
4. run the test_model.py file to load the trained model and output the accuracy on the test data. This will also create an image of the confusion matrix for the test result.

