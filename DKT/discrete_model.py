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

class TestCallback(Callback):

    def __init__(self, test_data = [[],[]]):
        self.x_test, self.y_test = test_data

    def on_epoch_begin(self, epoch, logs={}):

        y_pred = self.model.predict(self.x_test)
        avg_rmse, avg_acc = self.rmse_masking(self.y_test, y_pred)
        print('\nTesting avg_rmse: {}\n'.format(avg_rmse))
        print('\nTesting avg_acc: {}\n'.format(avg_acc))


    def rmse_masking(self, y_true, y_pred):
        #mask_matrix = np.sum(self.y_test_order, axis=2).flatten()
        num_users, max_sequences = np.shape(self.x_test)[0], np.shape(self.x_test)[1]
        #we want y_pred and y_true both to be matrix of 2 dim.
        if len(y_pred.shape) and len(y_true.shape) == 3:
            y_pred = y_pred.flatten()
            y_true = y_true.flatten()
        rmse = []
        acc = []
        padding_num = 0
        for user in range(num_users):
            diff_sq, response, correct = 0, 0, 0
            for i in range(user * max_sequences, (user + 1) * max_sequences):
                if y_true[i] == -1:
                    continue
                if y_true[i] == 1 and y_pred[i] >0.5:
                    correct += 1
                elif y_true[i] == 0 and y_pred[i] < 0.5:
                    correct += 1
                response += 1
                diff_sq += (y_true[i] - y_pred[i]) ** 2
            if response != 0:
                acc.append(correct/float(response))
                rmse.append(sqrt(diff_sq/float(response)))
        # print ('padding_num',padding_num)
        try:
            return sum(rmse)/float(len(rmse)), sum(acc)/float(len(acc))
        except:
            pdb.set_trace()
    '''
    def rmse_masking_on_batch(self, y_true, y_pred, y_order):
        num_users, max_responses = np.shape(y_order)[0], np.shape(y_order)[1]
        mask_matrix = np.sum(y_order, axis=2).flatten()
        #we want y_pred and y_true both to be matrix of 2 dim.
        if len(y_pred.shape) and len(y_true.shape) == 3:
            y_pred = y_pred.flatten()
            y_true = y_true.flatten()
        rmse = []
        acc = []
        padding_num = 0
        for user in range(num_users):
            diff_sq, response, correct = 0, 0, 0
            for i in range(user * max_responses, (user + 1) * max_responses):
                if mask_matrix[i] == 0:
                    break
                if y_true[i] == 1 and y_pred[i] >0.5:
                    correct += 1
                elif y_true[i] == 0 and y_pred[i] < 0.5:
                    correct += 1
                elif y_true[i] == -1:
                    padding_num += 1
                response += 1
                diff_sq += (y_true[i] - y_pred[i]) ** 2
            if response != 0:
                acc.append(correct/float(response))
                rmse.append(sqrt(diff_sq/float(response)))
        # print ('padding_num',padding_num)
        try:
            return rmse, acc
            # return sum(rmse)/float(len(rmse)), sum(acc)/float(len(acc))
        except:
            pdb.set_trace()
    '''       
            

class discrete_net():
    def __init__(self,batch_size, epoch, hidden_layer_size,input_dim):
        self.input_dim = input_dim # we don't need to specify input_dim here
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
                  callbacks = [ earlyStopping, \
                                TestCallback([self.val_index, \
                                self.val_response])],\
                                validation_data = (self.val_index,self.val_response), shuffle = True)
        
        
        
        
        
        
        