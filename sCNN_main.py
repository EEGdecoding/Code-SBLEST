import random
import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from braindecode.datautil.signal_target import SignalAndTarget
import logging
import sys
import os
from scipy.io import loadmat
from braindecode.models.deep4 import Deep4Net
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import (
    LossMonitor,
    MisclassMonitor,
    RuntimeMonitor,
)
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import BalancedBatchSizeIterator
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import set_random_seeds, np_to_var
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (
    bandpass_cnt,
    exponential_running_standardize,
)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne

log = logging.getLogger(__name__)

seed=20190706
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def run_exp(data_folder, model,  cuda):
    input_time_length = 750 # 1000
    max_epochs = 1600
    max_increase_epochs = 160
    batch_size = 60
    valid_set_fraction = 0.2
    # load data
    X = np.zeros([1])  # np.ndarray([])
    y = np.zeros([1])  # np.ndarray([])
    train_set = SignalAndTarget(X, y)
    test_set = SignalAndTarget(X, y)

    # Load train and test datasets from .mat files
    train_filename = 'Dataset2_L1_FootTongue_train.mat'
    test_filename = 'Dataset2_L1_FootTongue_test.mat'
    train_filepath = os.path.join(data_folder, train_filename)
    test_filepath = os.path.join(data_folder, test_filename)
    train = loadmat(train_filepath)
    test = loadmat(test_filepath)
    #

    train_set.X = np.transpose(train['X_train'], (2, 0, 1)).astype(np.float32)
    test_set.X = np.transpose(test['X_test'], (2, 0, 1)).astype(np.float32)
    train['Y_train'] = np.where(train['Y_train'] == -1, 0, train['Y_train'])
    test['Y_test'] = np.where(test['Y_test'] == -1, 0, test['Y_test'])
    train_set.y = train['Y_train'].astype(np.int64)
    test_set.y = test['Y_test'].astype(np.int64)
    train_set.y = train_set.y.reshape(np.size(train_set.y, 0))
    test_set.y = test_set.y.reshape(np.size(test_set.y, 0))

    #   split data into two sets
    train_set, valid_set = split_into_two_sets(
        train_set, first_set_fraction=1-valid_set_fraction)

    n_classes = 2
    n_chans = int(train_set.X.shape[1])
    input_time_length = train_set.X.shape[2]
    if model == "shallow":
        model = ShallowFBCSPNet(
            n_chans,
            n_classes,
            input_time_length=input_time_length,
            final_conv_length="auto",
        ).create_network()
    elif model == "deep":
        model = Deep4Net(
            n_chans,
            n_classes,
            input_time_length=input_time_length,
            final_conv_length="auto",
        ).create_network()
    if cuda:
        model.cuda()
    log.info("Model: \n{:s}".format(str(model)))

    optimizer = optim.Adam(model.parameters())

    iterator = BalancedBatchSizeIterator(batch_size=batch_size)

    stop_criterion = Or(
        [
            MaxEpochs(max_epochs),
            NoDecrease("valid_misclass", max_increase_epochs),
        ]
    )

    monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]

    model_constraint = MaxNormDefaultConstraint()

    exp = Experiment(
        model,
        train_set,
        valid_set,
        test_set,
        iterator=iterator,
        loss_function=F.nll_loss,
        optimizer=optimizer,
        model_constraint=model_constraint,
        monitors=monitors,
        stop_criterion=stop_criterion,
        remember_best_column="valid_misclass",
        run_after_early_stop=True,
        cuda=cuda,
    )
    exp.run()


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s : %(message)s',
        level=logging.DEBUG,
        stream=sys.stdout,
    )

    data_folder = 'C:/Users\Administrator\Desktop\Code-SBLEST-main'

    model = "shallow"  # 'shallow' or 'deep'
    cuda = True  # True False
    exp = run_exp(data_folder,  model, cuda)
    log.info("Last 5 epochs")
    log.info("\n" + str(exp.epochs_df.iloc[-5:]))
    Accuracy = 1-exp.epochs_df.iloc.obj.test_misclass[-1:]
    print('mean accuracy', Accuracy)