from keras.models import Model
from keras.layers import Input, Dropout, Masking, Dense, Embedding
from keras.layers import Embedding
from keras.layers.core import Flatten, Reshape
from keras.layers import LSTM
from keras.layers.recurrent import SimpleRNN
from keras.layers import merge
from keras.layers.merge import multiply
from keras.callbacks import EarlyStopping
from keras import backend as K
from theano import tensor as T
from theano import config
from theano import printing
from theano import function
from keras.layers import Lambda
import theano
import numpy as np
import pdb
from math import sqrt
from keras.callbacks import Callback



class discrete_net():
    def __init__(self,batch_size, epoch, hidden_layer_size):
        self.input_dim = 1 # the index per student is a vector
        self.output_dim = 1 # the output response is 1/0
        self.batch_size = batch_size
        self.epoch = epoch
        self.hidden_layer_size = hidden_layer_size
        print ("Model Initialized")
    
    def custom_bce(self, y_true, y_pred):
        b = K.not_equal(y_true, -K.ones_like(y_true))
        b = K.cast(b, dtype='float32')
        losses = K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1) * K.mean(b, axis=-1)
        count =  K.not_equal(losses, 0).sum()
        return  losses.sum()/count

    def build(self, train_index, train_response, val_index, val_response):
        self.train_index = train_index
        self.train_response = train_response
        self.val_index = val_index
        self.val_response = val_response
        
        x = Input(batch_shape = (None, None, self.input_dim), name='x')
        masked = (Masking(mask_value= -1, input_shape = (None, None, self.input_dim)))(x)
        lstm_out = SimpleRNN(self.hidden_layer_size, input_shape = (None, None, self.input_dim), return_sequences = True)(masked)
        dense_out = Dense(self.output_dim, input_shape = (None, None, self.hidden_layer_size), activation='sigmoid')(lstm_out)
        earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')
        
        model = Model(inputs=x, outputs=dense_out)
        model.compile( optimizer = 'rmsprop',\
                       loss = self.custom_bce,\
                      metrics=['accuracy'])

        model.fit(self.train_index, self.train_response, batch_size = self.batch_size, \
                  epochs=self.epoch, \
                  callbacks = [ earlyStopping], \
                                validation_data = (self.val_index,self.val_response), shuffle = True)
        
        
        
        
        
        
        