import sys
from datetime import datetime
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation,Flatten
from keras.layers import Conv1D, MaxPooling1D,GlobalAveragePooling1D,GlobalMaxPooling1D
from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers.core import Dropout
from keras.layers.noise import GaussianDropout
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
from keras.layers.core import Lambda
from keras import backend as K
from dataGenerator import gen,printSeqNum,isCombination,balancingData
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from keras.layers.recurrent import LSTM, GRU


from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from sn_sp import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

#based on Keras 2.0.6
# data path,data path, training sample size, saved model path, optimizer

np.random.seed(1337)
print('Program started at: '+str(datetime.now()))

data_path=str(sys.argv[1])
ng_data_path=sys.argv[2]
sample_size=int(sys.argv[3])
save_path=str(sys.argv[4])
opt=str(sys.argv[5])
#opt='SGD'
print('data path',data_path,ng_data_path)
print('sample size',sample_size)
print('save model path',save_path)
print('optimizer',opt)

batch_size = min(128,int(sample_size/10))
nb_epoch = 500

#get data
print('Loading data...')
data=np.loadtxt(data_path)
data=data.reshape(data.shape[0],4,data.shape[1]/4)
fakeData=np.loadtxt(ng_data_path)
fakeData=fakeData.reshape(fakeData.shape[0],4,fakeData.shape[1]/4)

data,fakeData=balancingData(data, fakeData)

#data=np.transpose(data,axes=(0,2,1))
print('Data has shape:',data.shape)
#fakeData=np.transpose(fakeData, axes=(0,2,1))
print('Fake data has shape:',fakeData.shape)

X=np.concatenate((data,fakeData))
Y=[1]*data.shape[0]+[0]*fakeData.shape[0]
Y=np.array(Y)

#shuffle data
idx=np.random.choice(X.shape[0],sample_size,replace=False)
X=X[idx]
Y=Y[idx]
#split
split_v=int(0.8*X.shape[0])
split_t=int(0.9*X.shape[0])


if opt=='SGD':
    opt=SGD(lr=0.01, momentum=0.9)

model=Sequential()

model.add(LSTM(32, input_shape=(4,100)))
model.add(Dropout(0.2))
#model.add(LSTM(64, return_sequences=False))
#model.add(Dropout(0.2))

model.add(Dense(output_dim=128, init='glorot_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.8))

model.add(Dense(output_dim=1))
model.add(Activation('sigmoid'))

'''
model.add(Convolution1D(input_dim=4, input_length=100, nb_filter=64, filter_length=16, border_mode="valid", subsample_length=1, init='glorot_normal'))
model.add(Flatten(input_shape=data.shape[1:]))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
'''
print model.summary()
checkpointer=ModelCheckpoint(filepath=save_path,save_best_only=True,verbose=1)
earlystopper=EarlyStopping(monitor='val_loss',patience=50,verbose=1)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X[:split_v], Y[:split_v], batch_size=batch_size, epochs=nb_epoch, verbose=2, validation_data=(X[split_v:split_t],Y[split_v:split_t]), callbacks=[checkpointer,earlystopper], shuffle=True)
model=load_model(save_path)
score=model.evaluate(X[split_t:], Y[split_t:], batch_size=batch_size, verbose=0)
print(score)

test_prediction=model.predict(X[split_t:])
auc_test=roc_auc_score(Y[split_t:], test_prediction)
print 'Test auc score: {}'.format(auc_test)

y_test = Y[split_t:]
predrslts = test_prediction
X_test = X[split_t:]

auc = roc_auc_score(y_test, predrslts)
predrslts_class = model.predict_classes(X_test, verbose=1)
mcc = matthews_corrcoef(y_test, predrslts_class)
acc = accuracy_score(y_test, predrslts_class)
sn, sp = SensitivityAndSpecificity(predrslts, y_test)
f1 = f1_score(y_test, predrslts_class, average='binary')


print 'auc:', auc
print 'mcc:', mcc
print 'acc:', acc
print 'sn:', sn
print 'sp:', sp
print 'f1:', f1

#find out what's wrong:
test_class=model.predict_classes(X[split_t:],verbose=0).reshape(-1)
print 'Test class result has shape {}'.format(test_class.shape)

'''
test_Y=Y[split_t:]
test_X=X[split_t:]
for i in range(len(test_Y)):
    if test_Y[i]<>test_class[i]:
        #seqNum=printSeqNum(np.transpose(test_X[i,:,28:35],axes=(1,0)))
        print'Not correctly predicted,true class is {}, predicted class is {}'.format(test_Y[i],test_class[i])
        print test_X[i,:,28],test_X[i,:,30],test_X[i,:,32],test_X[i,:,34]
'''

