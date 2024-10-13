import torch
import argparse
import os
from make_datasets import make_dataset
from DNA_Dataset_creator_txt import DNA_dataset_creator
from DNA_Trainer import DNA_model
from DNA_Data_Preprocessing import test_and_plot
from Bio import SeqIO
import subprocess

def count_sequences(file):
    count = 0
    for record in SeqIO.parse(file, "fasta"):
        count += 1
    return count

def vts_checker(vts):
    print(vts)
    vs = int(vts[0]) 
    ts = int(vts[1])
    if (not (vs > 0 and vs < 100)) or (not (ts > 0 and ts < 100)):
        raise argparse.ArgumentTypeError("Input must be an integer between 0 and 100")
    if 100 - vs - ts <= 0:
        raise argparse.ArgumentTypeError("No Training Data Remaining")
        
    return [vs, ts]

def prepare_dataset(files, class_names, gene_file, organ_file, method=1, gene_length=1000, window=False):
    DNA_dataset = dict()
    protien_dataset = []
    for i, directory in enumerate(files):
        path = os.path.dirname(os.path.abspath(__file__)) + '/' + directory
        DNA_dataset[directory] = DNA_dataset_creator(path, gene_length, method, class_names, i, gene_file, organ_file, window)
    
    return protien_dataset, DNA_dataset

if __name__ == '__main__':

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


    cmdline_parser = argparse.ArgumentParser('train')

    cmdline_parser.add_argument('-f', '--train_files',
                                default="",
                                nargs="+",
                                help='Name of file',
                                required=False
                                )
    cmdline_parser.add_argument('-m', '--method',
                                default=1,
                                nargs='?',
                                type=int,
                                choices=[1, 2, 3],
                                help='Preprocessing method (integer expected), where options are: '
                                    '1 for padding, 2 for contigs assembly, 3 for sliding window, \n If empty padding will be used.',
                                required=False
                                )
    cmdline_parser.add_argument('-l', '--gene_length',
                            default=1000,
                            nargs="?",
                            help='length of the gene for training and testing, default is 1000',
                            type=int,
                            required=False
                            )
    cmdline_parser.add_argument('-w', '--window',
                        default=1000000,
                        nargs="?",
                        help='size of the window, only works if sliding window is selected for preprocessing method. in case of not valid window size (e.g. window size larger the gene length), then the window size will set to gene_length/10',
                        type=int,
                        required=False
                        )
    cmdline_parser.add_argument('-vts', '--split',
                                default=[10, 10],
                                nargs=2,
                                help='validation and test percentage respectively',
                                required=False,
                                type=int
                                )
    cmdline_parser.add_argument('-predict', '--predict_mode',
                                action='store_true',
                                help='Use this flag if there is trained models for prediction',
                                required=False
                                )
    cmdline_parser.add_argument('-p', '--protein_sequences',
                                default="",
                                action='store_true',
                                help='Provide protein sequences only',
                                required=False
                                )
    cmdline_parser.add_argument('-o', '--outpath',
                                default="./",
                                help='Provide provide output path',
                                type=str,
                                required=False
                                )
    
    args, unknowns = cmdline_parser.parse_known_args()
    
    
    try:
        vs, ts = vts_checker(args.split)
    except argparse.ArgumentTypeError as e:
        cmdline_parser.error(str(e))
    
    print("using cuda: " + str(torch.cuda.is_available()))

    class_names = args.train_files
    classes = len(class_names)
    files_train = args.train_files
    method = args.method
    gene_length = args.gene_length

    if method == 3:
        if args.window > 0:
            window = args.window if args.window <= gene_length else int(gene_length/10)
        else:
            window = int(gene_length/10)
    else:
        window = False


    if not args.predict_mode:
        if not args.protein_sequences:
            DNA_train = dict()

            gene_file_train = 'gene_'+str(gene_length)+'_no_duplicates.txt'
            organ_file_train = 'organ_'+str(gene_length)+'_no_duplicates.txt'
            gene_file_test = 'gene_'+str(gene_length)+'_no_duplicates_test.txt'
            organ_file_test = 'organ_'+str(gene_length)+'_no_duplicates_test.txt'
            gene_file_val = 'gene_'+str(gene_length)+'_no_duplicates_val.txt'
            organ_file_val = 'organ_'+str(gene_length)+'_no_duplicates_val.txt'

            train_files = []
            test_files = []
            val_files = []

            # Remove the output file if it already exists
            if os.path.exists(gene_file_train):
                os.remove(gene_file_train)

            # Remove the output file if it already exists
            if os.path.exists(organ_file_train):
                os.remove(organ_file_train)

                # Remove the output file if it already exists
            if os.path.exists(gene_file_test):
                os.remove(gene_file_test)

                # Remove the output file if it already exists
            if os.path.exists(organ_file_test):
                os.remove(organ_file_test)

                # Remove the output file if it already exists
            if os.path.exists(gene_file_val):
                os.remove(gene_file_val)

                # Remove the output file if it already exists
            if os.path.exists(organ_file_val):
                os.remove(organ_file_val)

            train_files, DNA_train = prepare_dataset(files_train, class_names, gene_file_train, organ_file_train, method, gene_length, window)

            path = args.outpath + "/trainingdata"

            for key, value in DNA_train.items():
                make_dataset(value[0], value[1], path, vs/100, ts/100, 'train')


            outpath = args.outpath
            X_train, Y_train, X_val, Y_val, X_test, Y_test, number_subsequences = test_and_plot(path, outpath, 'test_model', do_shrink_timesteps = False)
            DNA_model(X_train, X_val, Y_train, Y_val, outpath, sampleSize=1, nodes=150, suffix="test_model", epochs=100, dropout=0.2)

    else:

        from DNA_Prediction_Preprocessing import test_and_plot
        from DNA_Predictor import predict

        X_test, Y_test, number_sequences = test_and_plot(args.outpath, args.outpath, "test_model")
        accuracy = predict(X_test, Y_test, model_path=args.outpath)
        print("The test accuracy is: " + str(accuracy))


