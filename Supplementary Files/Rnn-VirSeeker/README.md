This directory contains the script to train and test the Rnn-VirSeeker model, along with scripts to create the dataset. To run the script and train a model, use the following command:
```
python train.py
```
The model was trained on 452,608 sequences of length 500, which is the same training data used to train HVSeeker. Additionally, the training model is composed of 256 LSTM units, with a small learning rate of 0.00001    
The training and testing process undergoes the following steps:  
1. run the rnn_seeker_dataset_creator.py file. This will prepare the dataset in the same format that is accepted by Rnn-VirSeeker.
2. run the rnn_seeker_output_creator.py file. This will prepare the output in the same format that is accepted by Rnn-VirSeeker.
3. run the train.py file for training. The training will go through 100 epochs with a batch size of 256.
4. save the model in the same directory.
5. run the test.py file to load the trained model and output the accuracy on the test data. This will also create an image of the confusion matrix for the test result.
