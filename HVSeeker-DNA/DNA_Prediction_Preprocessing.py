import pandas as pd
import warnings
import numpy as np
from DNA_Predictor import predict
from sklearn.preprocessing import LabelEncoder
warnings.simplefilter(action='ignore', category=FutureWarning)

import DataParsing_main


class ReverseLabelEncoder(LabelEncoder):
    def fit(self, y):
        self.classes_ = np.array(sorted(np.unique(y), reverse=True))
        return self
    def get_mapping(self):
        return {label: idx for idx, label in enumerate(self.classes_)}

def test_and_plot(inpath, outpath, suffix, 
                  do_shrink_timesteps=False, online_training=False, 
                  one_hot_encoding=True, repeat=True, use_repeat_spacer=False, val_size=0.3,
                  input_subSeqlength=0, nodes=32, faster=False,
                  **kwargs):
    """
    1. gets settings and prepare data
    2. saves settings
    3. starts training
    4. saves history
    5. plots results
    :return:
    """

    # GET SETTINGS AND PREPARE DATA
    global X_test, Y_test, batch_size, SEED, y_encoder, number_subsequences

    batch_size = 32
    SEED = 42
    directory = ''

    maxLen, Y_test, mapping = use_data_nanocomb(directory=inpath, one_hot_encoding=one_hot_encoding, repeat=repeat,
                               use_spacer=use_repeat_spacer,
                               online=online_training, unbalanced=False, maxLen=kwargs.get("maxLen", 0))


    if do_shrink_timesteps:
        X_test, Y_test, number_subsequences = DataParsing_main.shrink_timesteps(X_test, Y_test, input_subSeqlength)
    else:
        number_subsequences = 1

    print(f"number of subsequences used per sequence: {number_subsequences}")
    """to limit the training on specified classes/hosts"""


    return X_test, Y_test, number_subsequences, mapping

def use_data_nanocomb(directory, one_hot_encoding=True, repeat=True, use_spacer=True, maxLen=0, online=False,
                      unbalanced=False):
    """
    to use the nanocomb exported data
    """

    Y_test_old = pd.read_csv(directory + '/Y_test.csv', delimiter='\t', dtype='str', header=None)[1].values
    print(type(Y_test_old))
    X_test_old = pd.read_csv(directory + '/X_test.csv', delimiter='\t', dtype='str', header=None)[1].values


    create_val = False


    if one_hot_encoding:
        global X_test, Y_test
        if maxLen <= 0:
            length = []
            x_sets = [X_test_old]

            for X in x_sets:
                for i in X:
                    length.append(len(i))
            length.sort()

            print(f"shortest sequence = {length[0]}")
            if maxLen == -1:
                maxLen = length[0]
            else:
                maxLen = length[int(len(length) * 0.95)]
            print(f"maxLen = {maxLen}")

        X_test = DataParsing_main.encode_string(maxLen=maxLen, x=X_test_old, repeat=repeat, use_spacer=use_spacer)

    else:
        X_test = X_test_old


    
    encoder = ReverseLabelEncoder()
    encoder.fit(Y_test_old)
    # print(encoder.classes_)
    # print(encoder.transform(encoder.classes_))

    Y_test = encoder.transform(Y_test_old)
            
            
    #unique_labels = np.unique(Y_test_old)
    #mapping = {}
    #for enum,key in enumerate(unique_labels):
    #    if 'Bacteria' in key:
    #        mapping[key] = 1
    #    else:
    #        mapping[key] = 0
    

    #mapping = {'Bacteria': 0, 'Phage': 1}  # Adjust this to your needs
    #vectorized_mapping = np.vectorize(lambda x: mapping.get(x, x))
    # Apply the mapping to the DataFrame
    #Y_test = vectorized_mapping(Y_test_old)


    return maxLen, Y_test, encoder.get_mapping()

