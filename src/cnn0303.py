from __future__ import division

import os
os.environ['THEANO_FLAGS'] = "device=gpu"
import sys
sys.setrecursionlimit(15000)

import numpy as np
import h5py
import scipy.io
np.random.seed(1337) # for reproducibility
np.set_printoptions(threshold=np.inf)  
from keras import backend as K
from sklearn.metrics import f1_score
from keras.models import load_model, Model
from keras.preprocessing import sequence
from keras.optimizers import RMSprop, SGD
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.layers.local import LocallyConnected1D
from keras.layers.pooling import AveragePooling1D

from keras.regularizers import l1, l2
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.wrappers import Bidirectional
#from seya.layers.recurrent import Bidirectional
#from keras.utils.layer_utils import print_layer_shapes

from keras.layers.normalization import BatchNormalization
#from residual_blocks import building_residual_block
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
from sklearn.cross_validation import train_test_split

from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

import pandas
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sn_sp import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

#from keras.utils import plot_model
#from keras.utils.visualize_util import plot

import pandas as pd
from vis.visualization import visualize_saliency
from vis.utils import utils
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

import pickle

from keras.utils import plot_model
from vis.visualization import visualize_saliency
from matplotlib import pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
from vis.utils import utils
from pylab import plot, show
#import cv2
#from attentionMap import visualize_saliency, get_num_filters, visualize_cam

saliency_img_file = "saliency_map.png"
original_img_file = "p4_original.png"

print 'loading data'

model_file_name = "./cnn0303.hdf5";
result_file_name = "./cnn0303.res";


print model_file_name;
print result_file_name;

trainmat = h5py.File('./train.hdf5', 'r')
validmat = h5py.File('./valid.hdf5', 'r')
testmat = h5py.File('./test.hdf5', 'r')

X_train = np.transpose(np.array(trainmat['x_train']),axes=(0,2,1))
y_train = np.array(trainmat['y_train'])

X_test = np.transpose(np.array(testmat['x_test']),axes=(0,2,1))
y_test = np.array(testmat['y_test'])

#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state=7)


X_valid = np.transpose(np.array(validmat['x_valid']),axes=(0,2,1))
y_valid = np.array(validmat['y_valid'])


NUM_FILTER1 = 16
INPUT_LENGTH = 134 # 162 #688  #162

print 'building model'

model = Sequential()

##first conv layer
model.add(Convolution1D(input_dim=4,
                        input_length=INPUT_LENGTH,
                        nb_filter=NUM_FILTER1,
                        filter_length=8,
                        border_mode="valid",
                        activation='relu',
                        subsample_length=1, init='glorot_normal'))

#model.add(MaxPooling1D(pool_length=2, stride=2))
model.add(Dropout(0.4))


##residual blocks
input_length2, input_dim2 = model.output_shape[1:]
nb_filter2 = 32
filter_length = 2
subsample = 1


model.add(Convolution1D(input_dim = input_dim2,
                        input_length = input_length2,
                        nb_filter = nb_filter2,
                        filter_length = 6,
                        border_mode = "valid",
                        activation='relu',
                        subsample_length = 1, init = 'glorot_normal'))

#model.add(MaxPooling1D(pool_length=2, stride=2))
model.add(Dropout(0.3))




input_length3, input_dim3 = model.output_shape[1:]
model.add(Convolution1D(input_length = input_length3, input_dim = input_dim3,
                        nb_filter = 64, filter_length = 4,
			border_mode = "valid",
                        activation='relu',
                        subsample_length = 1, init = 'glorot_normal'))
                        

model.add(MaxPooling1D(pool_length=2, stride=2))
model.add(Dropout(0.3))


#model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(output_dim=64, init='glorot_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(output_dim=1, name='preds'))
model.add(Activation('sigmoid'))

#print 'compiling model'
#model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary")


#model = load_model(model_file_name)

print 'compiling model'
sgd = SGD(lr=0.001, momentum=0.9, decay=1e-5, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
# Compile model
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print 'running at most 60 epochs'

checkpointer = ModelCheckpoint(filepath=model_file_name,monitor='val_loss', verbose=1, save_best_only=True, mode='min')
earlystopper = EarlyStopping(monitor='val_loss', patience=200, verbose=1)


#tresults = model.evaluate(np.transpose(testmat['testxdata'],axes=(0,2,1)), testmat['testdata'],show_accuracy=True)
#tresults = model.evaluate(X_test, Y_test, verbose = 0)

print model.summary()
result = model.fit(X_train, y_train, batch_size=128, nb_epoch=10000, initial_epoch=5, shuffle=True, validation_data=(X_valid, y_valid), callbacks=[checkpointer,earlystopper])

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

save_obj(result.history, "run01.pkl")

def load_result(files):
    labels = ["acc", "loss", "val_loss", "val_acc"]
    ret = {"acc":[], "loss":[], "val_loss":[], "val_acc":[],}
    for path in files:
        result = load_obj(path)
        for l in labels:
            ret[l] = ret[l] + result[l]
    return ret

result = load_result(["run01.pkl"])

fig = plt.figure(figsize=(20,4))
plt.subplot("121")
plt.plot(range(len(result["loss"])), result["loss"], label="train_loss")
plt.plot(range(len(result["val_loss"])), result["val_loss"], label="val_loss")
plt.xlabel("epochs")
plt.ylabel("logloss")

plt.subplot("122")
plt.plot(range(len(result["acc"])), result["acc"], label="train_acc")
plt.plot(range(len(result["val_acc"])), result["val_acc"], label="val_acc")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.savefig('epochs01.png')

'''
# list all data in history
print(history.history.keys())


# summarize history for accuracy
plt.figure(0)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('acc2.png')

# summarize history for loss
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss2.png')
'''

#plot(model, to_file='model2.png', show_shapes=True)

model.layers[1].get_weights()
tresults = model.evaluate(X_test, y_test)

print 'predicting on test sequences'
model.load_weights(model_file_name)
predrslts = model.predict(X_test, verbose=1)



from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations
'''
#from matplotlib import pyplot as plt
#%matplotlib inline
plt.rcParams['figure.figsize'] = (4, 162)

# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'preds')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

# This is the output node we want to maximize.
filter_idx = 0
img = visualize_activation(model, layer_idx, filter_indices=filter_idx)
outf = np.expand_dims(img, axis=0)
plt.imsave(saliency_img_file, img)


for idx, layer in enumerate(model.layers) :
    print(idx)
    print(layer.name)
    print('~~~~')

layer_name = 'conv1d_1'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

#indices = [6, 8, 21, 38, 416, 423, 425]

#data = pd.read_csv(data_file, nrows = np.max(indices) + 1)
#x = X_test.reshape((data.shape[0], size, size, 1))
#imgs = X_train[indices]
x1 = X_test.reshape((X_test.shape[0], 1, 162, 4))
indices = [6, 8, 21, 38, 416, 423, 425]
imgs = x1[indices]


vis_images = []
heatmaps = []
for img in imgs:
    #seed_img = X_test[np.random.randint(i*5, 600)]
    #x = np.expand_dims(seed_img, axis=0)
    #x = preprocess_input(seed_img)
    pred_class = np.argmax(model.predict(img))
    # Here we are asking it to show attention such that prob of `pred_class` is maximized.
    heatmap = visualize_saliency(model, layer_idx, [pred_class], img)
    heatmaps.append(heatmap)

plt.axis('off')
plt.imsave(saliency_img_file, utils.stitch_images(heatmaps))
'''
#plt.title('Saliency map')
#plt.savefig('saliency_map_cnn')

#cv2.imwrite(saliency_img_file, utils.stitch_images(heatmaps, cols=2))
#cv2.imwrite(saliency_img_file, utils.stitch_images(heatmaps))
'''
#plt.axis('off')
#plt.imshow(utils.stitch_images(heatmaps))
plt.figure(0)
plt.imshow(utils.stitch_images(heatmaps))
plt.title('Saliency map')
plt.savefig('saliency_map_cnn')

for img, pc in zip(imgs, predrslts):
	heatmap = visualize_saliency(model, len(model.layers) - 1, [pc], img)
	heatmaps.append(heatmap)

cv2.imwrite(saliency_img_file, utils.stitch_images(heatmaps, cols=7))
cv2.imwrite(original_img_file, utils.stitch_images(list(imgs), cols=7))
'''
auc = roc_auc_score(y_test, predrslts)
predrslts_class = model.predict_classes(X_test, verbose=1)
mcc = matthews_corrcoef(y_test, predrslts_class)
acc = accuracy_score(y_test, predrslts_class)
sn, sp = SensitivityAndSpecificity(predrslts, y_test)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predrslts)
precision, recall, thresholds1 = precision_recall_curve(y_test, predrslts)

#print auc(false_positive_rate, true_positive_rate)
print tresults
print 'auc:', auc
print 'mcc:', mcc
print 'acc:', acc
print 'sn:', sn
print 'sp:', sp
f1 = f1_score(y_test, predrslts_class, average='binary')
print 'f1:', f1
#print 'fpr:', false_positive_rate
#print 'tpr:', true_positive_rate
'''
fig = plt.figure()
#fig.savefig('temp.png', dpi=fig.dpi)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label = 'AUC = %0.2f' % auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.show()
fig.savefig('temp2.png', dpi=fig.dpi)
'''

fw = open(result_file_name, 'w')
fw.write('\t'.join(['tresults', 'auc', 'mcc', 'acc', 'sn', 'sp']) +'\n')
fw.write('\t'.join([str(tresults), str(auc), str(mcc), str(acc), str(sn), str(sp)]) +'\n')
fw.close();
'''
fwr = open("cpp-fpr.txt", 'w')
fwr.write(str(false_positive_rate))
fwr.close();


fwr = open("cpp-tpr.txt", 'w')
fwr.write(str(true_positive_rate))
fwr.close();
'''
fwr = open("cnn0303-fpr.txt", 'w')
fwr.write(str(false_positive_rate))
fwr.close();


fwr = open("cnn0303-tpr.txt", 'w')
fwr.write(str(true_positive_rate))
fwr.close();


fwr = open("cnn0303-precision.txt", 'w')
fwr.write(str(precision))
fwr.close();


fwr = open("cnn0303-recall.txt", 'w')
fwr.write(str(recall))
fwr.close();


