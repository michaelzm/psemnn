# -*- coding: utf-8 -*-
"""new_net_lstm.ipynb
"""

import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, SpatialDropout1D, GRU
from tensorflow.keras.utils import to_categorical
import zipfile
import pickle
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
#from tensorflow.keras import backend as keras
#from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf
import time
from dataset_joiner import DatasetWorker, VocabularyWorker
from performance import PerformanceViewer, TrainingEval
from sklearn.model_selection import train_test_split

np.random.seed(1234567890)

# import data
# specify the type of information which shall be extracted
#extraction_of = 'contexts'
extraction_of = 'sentiments'
#extraction_of = 'aspects'

#sentiment, aspect oder modifier -> diese drei braucht man
#extraktion von polarität nicht gefragt


# specify filenames in the next line
if extraction_of in ['contexts']:
    filename = r'data_laptop_ctxt.json'
elif extraction_of in ['sentiments','aspects']:
    filename = r'D:/Uni/Master/3. Semester/Praxisseminar/data_laptop_absa.json'

## in this example, we use the glove word embeddings as input for the neural network
## download glove.42B.300d.txt from http://nlp.stanford.edu/data/glove.42B.300d.zip
filename_embedding_zip = r'C:/Users/Niklas/Downloads/glove.42B.300d.zip' # folder of downloaded glove zip file
## specify folder where to store the glove embeddings
filepath_embedding = filename_embedding_zip.replace('.zip','')
## unzip and save glove to a folder manually or with the next lines
if not os.path.exists(filepath_embedding):
    with zipfile.ZipFile(filename_embedding_zip,"r") as zip_ref:
        zip_ref.extractall(filepath_embedding)
os.listdir(filepath_embedding)[0]
filename_embedding = filepath_embedding + '/' + os.listdir(filepath_embedding)[0]


with open(filename,'r', encoding='utf8') as infile:
    example_data = json.load(infile)

max_seq_length = 100
ds = DatasetWorker(example_data)
ds.applyPreprocessing()
# we let the nn use 20% of train data to validate and hold back 10% for final eval 
# (data the net does not see while training)
ds.setTrainTestSplitRatio(0.9)
#options for splitDataset = all_agree, one_agrees, every_review
ds.splitDataset("every_review")
ds.buildDatasetSequence(max_seq_length)
ds.describe()

#build vocab and add embedding
vw = VocabularyWorker()
vw.buildVocabulary(ds.dataset)
vw.buildEmbedding(ds.train_labels)

# Convert data to Input format for neural network
x_train, y_train = vw.convert_tokens_labels_list_to_ids_list(ds.train_tokens, ds.train_labels, max_seq_length)
x_test, y_test = vw.convert_tokens_labels_list_to_ids_list(ds.test_tokens, ds.test_labels, max_seq_length)

#make classes cateogrical
y_train = to_categorical(y_train, num_classes = vw.n_tags)
y_test = to_categorical(y_test, num_classes = vw.n_tags)


# the following two layers should not be changed.
input_layer = Input(shape=(max_seq_length,))
embedding_layer = Embedding(vw.vocab_size, 300, weights=[vw.embedding_vectors], input_length=max_seq_length)(input_layer)

#lstm_layer = SpatialDropout1D(0.4)(embedding_layer)
#lstm_layer = Dropout(0.1)(embedding_layer)
#lstm_layer = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(lstm_layer)

gru_layer = Dropout(0.1)(embedding_layer)
gru_layer = Bidirectional(GRU(units=100, return_sequences=True, recurrent_dropout=0.1))(gru_layer)
# here, attention models have to be implemented in this model
#nur bestimmten wörtern aufmerksamkeit geben
# ...

# this last layer can/should be modified
output_layer = TimeDistributed(Dense(vw.n_tags, activation="softmax"))(gru_layer)
    
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss="categorical_crossentropy",
              optimizer='adam', 
              metrics=[tf.keras.metrics.Precision(), 
                       tf.keras.metrics.Recall(),
                       "accuracy"])
model.summary()


performance = PerformanceViewer()
evaluate_callback = TrainingEval(model, x_test, y_test, vw, ds, performance)

"""## Model fit"""

#reset preciously saved performance data
performance.resetHistory()

'''
#GridSearch
batches = [32, 64, 128]
epochs = [3, 5, 7]
accuracy = []
val_accuracy=[]
round=0


for batch_size in batches:
    for epoch_size in epochs:
        # the following two layers should not be changed.
        input_layer = Input(shape=(max_seq_length,))
        embedding_layer = Embedding(vw.vocab_size, 300, weights=[vw.embedding_vectors], input_length=max_seq_length)(input_layer)

        #lstm_layer = SpatialDropout1D(0.4)(embedding_layer)
        lstm_layer = Dropout(0.1)(embedding_layer)
        lstm_layer = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(lstm_layer)
        # here, attention models have to be implemented in this model
        #nur bestimmten wörtern aufmerksamkeit geben
        # ...

        # this last layer can/should be modified
        output_layer = TimeDistributed(Dense(vw.n_tags, activation="softmax"))(lstm_layer)
    
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
        history= model.fit(x_train, 
                  y_train, 
                  batch_size=batch_size, 
                  validation_split=0.2,
                  verbose=1,
                  #callbacks=[evaluate_callback],
                  epochs=epoch_size)
        accuracy.append(history.history['accuracy'])
        val_accuracy.append(history.history['val_accuracy'])
        round+=1
        print(round)

df=pd.DataFrame(accuracy)
df2=pd.DataFrame(val_accuracy)
mean_accuracy= df.mean(axis=1, skipna=True)
mean_val_accuracy= df2.mean(axis=1, skipna=True)     
   '''

#looking at previous history, data suggests that after 3 epochs, prediction quality falls off
# fit model on train data
history = model.fit(
    x_train, y_train,
    batch_size=32,
    validation_split = 0.2,
    verbose = 1,
    callbacks = [evaluate_callback],
    #validation_data=(x_test, y_test),
    epochs=5)

"""## Model evaluation"""

#performance.evalModelTrainDataClass()

#performance.evalModelTrainData()

#performance.basicEval(history)

performance.classicEval(model, ds,vw,x_test,y_test)

sen_matrix=performance.ClassificationReview(model, ds,vw,x_test,y_test)

wrong_sen = sen_matrix[sen_matrix["prediction"]=="False"] 

i_s_matrix = sen_matrix[sen_matrix["true_class"]=="I_S"]
i_s_matrix.sort_values(by=["prediction"], inplace=True) 

b_s_matrix = sen_matrix[sen_matrix["true_class"]=="B_S"]
b_s_matrix.sort_values(by=["prediction"], inplace=True)             
                
