# tRNA-DL

We proposed a new computational approach based on deep neural networks, to predict tRNA gene sequences. We designed and investigated various deep neural network architectures. We used tRNA sequences as positive samples and the false positive tRNA sequences predicted by tRNAscan-SE in coding sequences as negative samples, to train and evaluate the proposed models by comparison with the conventional machine learning methods and popular tRNA prediction tools. Using one-hot encoding method, our proposed models can extract features without involving extensive manual feature engineering. Our proposed best model outperformed the existing methods under different performance metrics.

The proposed deep learning methods can reduce the false positives output by the state-of-art tool tRNAscan-SE substantially. Coupled with tRNAscan-SE, it can serve as a useful complementary tool for tRNA annotation. The application to tRNA prediction demonstrates the superiority of deep learning in automatic feature generation for characterizing sequence patterns.



=======

DATA files are in fasta/ folder

The data processing codes are in data-processing/ folder

We use the following codes to split the postive and negative dataset into training, validation and test dataset respectively:
```
pos-split2.py
neg-split2.py
```

Then we transfer fasta data file into hdf5 format data for training and evaluating our model:
```
hdf5-train.py	
hdf5-valid.py	
hdf5-test.py	
hdf5-test2.py	
```


We devide positive samples into training set, sub-validation set, sub-test set and test set.
For examples, they are listed in order:
```
pos-train.fa	
pos-valid.fa
pos-test1.fa	
pos-test2.fa	
```
Negative samples are dealed with in the same way.

The one-hot encoding method is in the following python codes:
```
dna_io_1mer.py
```


Source codes include data processing part and model part, where data processing codes are in data-processing/ folder, models are list 
as follows:

| No.| Abbreviation | Prediction Deep Learning Model Architectures| 
| ------------- |:-------------:| -----:|
1 | CF | Conv1D + FC + SGD
2 | CCMF | Conv1D + Conv1D + MaxPool1D + FC + SGD
3 | CMCMF | Conv1D + MaxPool1D + Conv1D + MaxPool1D + FC + SGD
4 | CCMCAF | Conv1D + Conv1D + MaxPool1D + Conv1D + AvgPool1D + FC + SGD
5 | CMCMCMF | Conv1D + MaxPool1D + Conv1D + MaxPool1D + Conv1D + MaxPool1D + FC + SGD
6 | CCCMF | Conv1D + Conv1D + Conv1D + MaxPool1D + FC + SGD
7 | CMCMCF2 | Conv2D + MaxPool2D + Conv2D + MaxPool2D + Conv2D + FC + SGD
8 | CCCMF2 | Conv2D + Conv2D + Conv2D + MaxPool2D + FC + SGD
9 | LLLF | LSTM + LSTM + LSTM + FC + SGD
10 | CMBLF | Conv1D + MaxPool1D + BDLSTM + FC + RMSprop
11 | CCMBLF | Conv1D + Conv1D + MaxPool1D + BDLSTM + FC + RMSprop
12 | CMBGF | Conv1D + MaxPool1D + BDGRU + FC + RMSprop
13 | CCMBGF | Conv1D + Conv1D + MaxPool1D + BDGRU + FC + RMSprop


The following PYTHON codes shows the selected best performance CCMBLF model:
```py
def CCMBLF():
    model = Sequential()
    model.add(Convolution1D(input_dim=4,
                            input_length=134,
                            nb_filter=16,
                            filter_length=4,
                            border_mode="valid",
                            activation="relu",
                            subsample_length=1))

    model.add(Dropout(0.4))

    input_length2, input_dim2 = model.output_shape[1:]
    model.add(Convolution1D(input_dim=input_dim2,
                            input_length=input_length2,
                            nb_filter=64,
                            filter_length=4,
                            border_mode="valid",
                            activation="relu",
                            subsample_length=1))

    model.add(MaxPooling1D(pool_length=2, stride=2))

    input_length0, input_dim0 = model.output_shape[1:]

    model.add(Bidirectional(LSTM(input_dim=input_dim0, output_dim=64,
                                dropout_W=0.2, dropout_U=0.5,
                                # activation='relu',
                                return_sequences=True)))

    model.add(Flatten())

    model.add(Dense(output_dim=128))
    model.add(Activation('relu'))

    model.add(Dense(input_dim=128, output_dim=1))
    model.add(Activation('sigmoid'))

    return model;

model = CCMBLF()
```
