import argparse

from baselines.evolvegcn.constant import *


def get_link_prediction_args():
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument('--num_runs', type=int, default=5,
                        help='number of runs')
    parser.add_argument('--model_name', type=str, default='SGD-DyG',
                        help='name of the model')
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--patience", type=int, default=25,
                        help="number of epochs with no improvement on val before terminating")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="model learning rate")
    parser.add_argument("--lam", type=float, default=0.0005,
                        help="L2 regularization on feature tensor x")
    parser.add_argument("--tau", type=float, default=0.1,
                        help="cl loss weight")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="l2 norm for parameters")
    parser.add_argument('--seed', type=int, default=SEED,
                        help='random seed')
    parser.add_argument('--num_feature', type=int, default=8,
                        help='dimension of features')
    parser.add_argument('--layer', type=int, default=1,
                        help='Number of hidden units')
    parser.add_argument('--hidden_feature', type=int, default=16,
                        help='Number of hidden units')
    parser.add_argument("--dataset_name", type=str, default=DatasetType.WIKI_GL.name.lower(),
                        help='dataset to use for training', choices=[ds.name.lower() for ds in DatasetType])
    parser.add_argument("--bandwidth", type=int, default=20,
                        help='tensor graph convolution windows')
    parser.add_argument("--m_choice", type=int, default=2,
                        help='m matrix type')
    parser.add_argument("--cuda", type=bool, default=True,
                        help='enable cuda')
    parser.add_argument("--fft", type=bool, default=True,
                        help='enable fft module')
    parser.add_argument("--tgc_dropout", type=float, default=0.75,
                        help='tgc_dropout')
    parser.add_argument("--fft_dropout", type=float, default=0.75,
                        help='fft_dropout')
    parser.add_argument("--enable_cl", type=bool, default=True,
                        help='enable contrastive learning')
    parser.add_argument("--tensor_con", type=bool, default=True,
                        help='enable tensor graph convolution')
    args = parser.parse_args()

    return args
