import time
import os.path
from scipy.io import loadmat, savemat
import numpy as np
import logging
import sys
import random
from signal_target import SignalAndTarget,convert_numbers_to_one_hot
from splitters import split_into_two_sets

from keras.callbacks import EarlyStopping
import tensorflow as tf
import keras.backend as ka
import keras as k
import sys
from EEGModels import EEGNet, ShallowConvNet, DeepConvNet

time_start = time.time()

# TensorFlow configuration for GPU usage
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# Fix random seed
seed = 20190706
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

log = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                    level=logging.DEBUG, stream=sys.stdout)

# Data folder where the datasets are located
data_folder = 'C:/Users/Administrator/Desktop/Code-SBLEST-main'

# Fraction of data to be used as validation set
valid_set_fraction= 0.2

# Initialize train and test datasets
X = np.zeros([1]) # np.ndarray([])
y= np.zeros([1]) # np.ndarray([])
train_set = SignalAndTarget(X, y)
test_set = SignalAndTarget(X, y)

# Load train and test datasets from .mat files
train_filename = 'Dataset2_L1_FootTongue_train.mat'
test_filename = 'Dataset2_L1_FootTongue_test.mat'
train_filepath = os.path.join(data_folder, train_filename)
test_filepath = os.path.join(data_folder, test_filename)
train = loadmat(train_filepath)
test = loadmat(test_filepath)

# Prepare train and test datasets
label_1d_train = train['Y_train'].astype(np.int32)
label_1d_test = test['Y_test'].astype(np.int32)
train_set.y =convert_numbers_to_one_hot(label_1d_train)
test_set.y =convert_numbers_to_one_hot(label_1d_test)
train_set.X = np.transpose(train['X_train'], (2, 1, 0)).astype(np.float32)
test_set.X = np.transpose(test['X_test'], (2, 1, 0)).astype(np.float32)

# Split train set into train and validation set
train_set, valid_set = split_into_two_sets(
    train_set, first_set_fraction = 1 - valid_set_fraction
)

# Prepare data for model training and evaluation
X_train = np.expand_dims(train_set.X, axis=3)
X_validate = np.expand_dims(valid_set.X, axis=3)
X_test = np.expand_dims(test_set.X, axis=3)
Y_train = train_set.y
Y_valid = valid_set.y
Y_test = test_set.y

# Get number of channels and samples from input data
chans = X_train.shape[1]
samples = X_train.shape[2]
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Create and compile the EEGNet model
model = EEGNet(nb_classes=2, Chans=chans, Samples=samples,
               dropoutRate=0.5, kernLength=125, F1=8, D=2, F2=16,
               dropoutType='Dropout')

model.compile(loss='categorical_crossentropy', optimizer= 'adam',
              metrics=['accuracy'])

model.summary()

# Callbacks
early_stopping = EarlyStopping(
    monitor='loss', min_delta=0.0001,
    mode='min', patience=50, verbose=1,
    restore_best_weights=True)

# Train the model
fittedModel = model.fit(X_train, Y_train, batch_size=16, epochs=500,
                        verbose=2, validation_data=(X_validate, Y_valid),callbacks=[early_stopping]
                      )

# Predict the labels for the test set using the trained model and print the calculated classification accuracy
probs = model.predict(X_test)
preds = probs.argmax(axis=-1)
acc = np.mean(preds == Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))



