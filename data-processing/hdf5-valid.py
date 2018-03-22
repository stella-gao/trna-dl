import numpy as np
import h5py
import scipy.io
np.random.seed(1337) # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import RMSprop, SGD
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
#from keras.regularizers import l2, activity_l1
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils

from keras.layers.embeddings import Embedding

from dna_io_1mer import *


#file_name = 'human'
file_name_true = 'pos-valid'
file_name_false = 'neg-valid'



#seq_T_len = longest_seq_len(file_name_true+'.fa')
#seq_F_len = longest_seq_len(file_name_false+'.fa')


#max_len = max(seq_T_len, seq_F_len)
max_len = 134

seq_vecs_T = hash_sequences_1hot(file_name_true + '.fa', max_len)
seq_vecs_F = hash_sequences_1hot(file_name_false + '.fa', max_len)

seq_headers_T = sorted(seq_vecs_T.keys())
seq_headers_F = sorted(seq_vecs_F.keys())


train_seqs = []
train_scores = []

for header in seq_headers_T:
    train_seqs.append(seq_vecs_T[header])
    train_scores.append([1])

for header in seq_headers_F:
    train_seqs.append(seq_vecs_F[header])
    train_scores.append([0])

'''
for i in range(1000):
    train_seqs[i] = np.hstack((train_seqs[i],tmp))
'''
# print train_seqs.shape
print "---------------------------------"


train_seqs = np.array(train_seqs)
train_scores = np.array(train_scores)

# print train_seqs.shape

import h5py
h5f = h5py.File('valid.hdf5', 'w')
h5f.create_dataset('x_valid', data=train_seqs)
h5f.create_dataset('y_valid', data=train_scores)
h5f.close()

