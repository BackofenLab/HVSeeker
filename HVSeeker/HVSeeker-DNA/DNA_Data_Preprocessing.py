import pandas as pd
import os
from logging import warning
import warnings
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.utils import class_weight as clw
from sklearn.model_selection import train_test_split
import DataParsing_main

def test_and_plot(inpath, outpath, suffix, 
                  do_shrink_timesteps=True, online_training=False, 
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
    global X_train, X_test, X_val, Y_train, Y_test, Y_val, batch_size, SEED, y_encoder, number_subsequences


    batch_size = 32
    SEED = 42
    directory = ''
    Y_train_old = pd.read_csv(inpath + '/Y_train.csv', delimiter='\t', dtype='str', header=None)[1].values
    Y_train, y_encoder = DataParsing_main.encode_string(y=Y_train_old)
    print(*(zip(y_encoder.transform(y_encoder.classes_), y_encoder.classes_)))
    Y_train_noOHE = [y.argmax() for y in Y_train]
    class_weight = clw.compute_class_weight(class_weight = 'balanced', classes = np.unique(Y_train_noOHE), y = Y_train_noOHE)
    unbalanced = any([i != class_weight[0] for i in class_weight])
    maxLen = use_data_nanocomb(directory=inpath, one_hot_encoding=one_hot_encoding, repeat=repeat,
                               use_spacer=use_repeat_spacer,
                               online=online_training, unbalanced=unbalanced, maxLen=kwargs.get("maxLen", 0))

    if len(X_val) == 0:
        print("make new val set")
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_size, random_state=SEED,
                                                          stratify=Y_train)
    else:
        print("val set already exits")

    if nodes < X_test.shape[-1]:
        warning("use at least as many nodes as number of hosts to predict, better twice as much")

    if do_shrink_timesteps:

        if not online_training:
            X_train, Y_train, number_subsequences = DataParsing_main.shrink_timesteps(X_train, Y_train,
                                                                                      input_subSeqlength)
        X_val, Y_val, number_subsequences = DataParsing_main.shrink_timesteps(X_val, Y_val, input_subSeqlength)
        X_test, Y_test, number_subsequences = DataParsing_main.shrink_timesteps(X_test, Y_test, input_subSeqlength)
    else:
        number_subsequences = 1

    print(f"number of subsequences used per sequence: {number_subsequences}")
    """to limit the training on specified classes/hosts"""

    # SAVE SETTINGS
    with open(inpath + '/' + suffix + "_config.txt", "w") as file:
        for i in list(locals().items()):
            if i == 'Y_train_noOHE':
                continue

            file.write(str(i) + '\n')
        if faster == True:
            file.write('(\'batchsize\', ' + str(batch_size * 16) + ')\n')
        elif type(faster) == int and faster > 0:
            file.write('(\'batchsize\', ' + str(batch_size * faster) + ')\n')
        else:
            file.write('(\'batchsize\', ' + str(batch_size) + ')\n')
        file.write('(\'SEED\', ' + str(SEED) + ')\n')
        file.write('(\'directory\', ' + str(directory) + ')\n')

    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, number_subsequences

def use_data_nanocomb(directory, one_hot_encoding=True, repeat=True, use_spacer=True, maxLen=0, online=False,
                      unbalanced=False):
    """
    to use the nanocomb exported data
    """

    Y_train_old = pd.read_csv(directory + '/Y_train.csv', delimiter='\t', dtype='str', header=None)[1].values
    Y_test_old = pd.read_csv(directory + '/Y_test.csv', delimiter='\t', dtype='str', header=None)[1].values
    X_train_old = pd.read_csv(directory + '/X_train.csv', delimiter='\t', dtype='str', header=None)[1].values
    X_test_old = pd.read_csv(directory + '/X_test.csv', delimiter='\t', dtype='str', header=None)[1].values
    


    create_val = False


    
    Y_val_old = pd.read_csv(directory + '/Y_val.csv', delimiter='\t', dtype='str', header=None)[1].values
    X_val_old = pd.read_csv(directory + '/X_val.csv', delimiter='\t', dtype='str', header=None)[1].values
    
    print(directory + '/Y_val.csv')

    print("loaded validation set from: " + directory + '/Y_val.csv')

    """
    except:
        print("create validation set from train")
        create_val = True

    if create_val:
        assert unbalanced == False, "an unbalanced training set needs a predefined validation set"
        X_train_old, X_val_old, Y_train_old, Y_val_old = train_test_split(X_train_old, Y_train_old, test_size=0.3,
                                                                          random_state=SEED,
                                                                          stratify=Y_train_old)
        create_val = False
    """
    if one_hot_encoding:
        global X_test, X_train, X_val, Y_test, Y_train, Y_val
        if maxLen <= 0:
            length = []
            x_sets = [X_test_old, X_train_old]
            if create_val == False:
                x_sets.append(X_val_old)

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
            


        X_train = DataParsing_main.encode_string(maxLen=maxLen, x=X_train_old, repeat=repeat, use_spacer=use_spacer,
                                                 online_Xtrain_set=online)
        X_test = DataParsing_main.encode_string(maxLen=maxLen, x=X_test_old, repeat=repeat, use_spacer=use_spacer)

        if create_val == False:
            X_val = DataParsing_main.encode_string(maxLen=maxLen, x=X_val_old, repeat=repeat, use_spacer=use_spacer)
    else:
        X_train = X_train_old
        X_test = X_test_old
        if create_val == False:
            X_val = X_val_old

    Y_train, y_encoder = DataParsing_main.encode_string(y=Y_train_old)
    Y_test = DataParsing_main.encode_string(y=Y_test_old, y_encoder=y_encoder)
    Y_val = DataParsing_main.encode_string(y=Y_val_old, y_encoder=y_encoder)
    return maxLen


