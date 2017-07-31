import keras
from keras.models import Model
from keras.layers import Input, Dropout, Masking, Dense, Embedding
from keras.layers import Embedding
from keras.layers.core import Flatten, Reshape
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
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

class TestCallback_no_y_order(Callback):

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

            
class TestCallback_y_order(Callback):

    def __init__(self, test_data = [[],[],[]]):
        self.x_test, self.y_test_order, self.y_test = test_data

    def on_epoch_begin(self, epoch, logs={}):

        y_pred = self.model.predict([self.x_test, self.y_test_order])
        avg_rmse, avg_acc = self.rmse_masking(self.y_test, y_pred)
        print('\nTesting avg_rmse: {}\n'.format(avg_rmse))
        print('\nTesting avg_acc: {}\n'.format(avg_acc))


    def rmse_masking(self, y_true, y_pred):
        mask_matrix = np.sum(self.y_test_order, axis=2).flatten()
        num_users, max_responses = np.shape(self.x_test)[0], np.shape(self.x_test)[1]
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
                if y_true[i] == -1:
                    continue
                #if mask_matrix[i] == 0:
                    ##break
                    #continue
                if y_true[i] == 1 and y_pred[i] >0.5:
                    correct += 1
                elif y_true[i] == 0 and y_pred[i] < 0.5:
                    correct += 1
                #elif y_true[i] == -1:
                    #padding_num += 1
                    #response -= 1
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
            
            
            
            

class eyetracking_net():
    def __init__(self,batch_size, epoch, hidden_layer_size,input_dim, output_dim, learning_rate, optimizer_mode, RNN_mode):
        self.input_dim = input_dim # we don't need to specify input_dim here
        self.output_dim = output_dim # the output response is 1/0
        self.batch_size = batch_size
        self.epoch = epoch
        self.hidden_layer_size = hidden_layer_size
        
        self.learning_rate = learning_rate
        self.optimizer_mode = optimizer_mode
        self.RNN_mode = RNN_mode
        
        '''Choose optimizer and learning rate'''
        if self.optimizer_mode == 'RMSprop':
            self.optimizer =keras.optimizers.RMSprop(lr= learning_rate)
            
        elif self.optimizer_mode == 'Adagrad':
            self.optimizer = keras.optimizers.Adagrad(lr=learning_rate)
            
        elif self.optimizer_mode == 'Adamax':
            self.optimizer = keras.optimizers.Adamax(lr=learning_rate)
        else:
            print('Lack choice of optimizer or learning_rate')
        
            
            
        print ("Model Initialized")
    
    def custom_bce(self, y_true, y_pred):
        b = K.not_equal(y_true, -K.ones_like(y_true))
        b = K.cast(b, dtype='float32')
        losses = K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1) * K.mean(b, axis=-1)
        count =  K.not_equal(losses, 0).sum()
        return  losses.sum()/count

    def build_no_y_order(self, train_index, train_response, val_index, val_response):
        self.train_index = train_index
        self.train_response = train_response
        self.val_index = val_index
        self.val_response = val_response
        
        x = Input(batch_shape = (None, None, self.input_dim), name='x')
        masked = (Masking(mask_value= -1, input_shape = (None, None, self.input_dim)))(x)
        #RNN_out = SimpleRNN(self.hidden_layer_size, input_shape = (None, None, self.input_dim), return_sequences = True)(masked)
        
        if self.RNN_mode == 'SimpleRNN':
            RNN_out = SimpleRNN(self.hidden_layer_size, input_shape =\
                                (None, None, self.input_dim), return_sequences = True)(masked)
        elif self.RNN_mode == 'LSTM':
            RNN_out = LSTM(self.hidden_layer_size, input_shape =\
                                    (None, None, self.input_dim), return_sequences = True)(masked)
        elif self.RNN_mode == 'GRU':
            RNN_out = GRU(self.hidden_layer_size, input_shape =\
                                        (None, None, self.input_dim), return_sequences = True)(masked)
        else:
            print('Sth wrong with the RNN_mode')
                

        dense_out = Dense(self.output_dim, input_shape = (None, None, self.hidden_layer_size), activation='sigmoid')(RNN_out)
        earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')
        
        model = Model(inputs=x, outputs=dense_out)
        model.compile( optimizer = 'rmsprop',\
                       loss = self.custom_bce,\
                      metrics=['accuracy'])

        model.fit(self.train_index, self.train_response, batch_size = self.batch_size, \
                  epochs=self.epoch, \
                  callbacks = [ earlyStopping, \
                                TestCallback_no_y_order([self.val_index, \
                                self.val_response])],\
                                validation_data = [self.val_index,self.val_response], shuffle = True)
        
    def build_y_order(self, train_index, train_order,train_response,\
                      val_index, val_order, val_response):
        self.train_index = train_index
        self.train_order = train_order
        self.train_response = train_response
        

        self.val_index = val_index
        self.val_order = val_order
        self.val_response = val_response
        
        
        x = Input(batch_shape = (None, None, self.input_dim), name='x')
        masked = (Masking(mask_value= -1, input_shape = (None, None, self.input_dim)))(x)
        #RNN_out = SimpleRNN(self.hidden_layer_size, input_shape = (None, None, self.input_dim), return_sequences = True)(masked)
        
        if self.RNN_mode == 'SimpleRNN':
            RNN_out = SimpleRNN(self.hidden_layer_size, input_shape =\
                                (None, None, self.input_dim), return_sequences = True)(masked)
        elif self.RNN_mode == 'LSTM':
            RNN_out = LSTM(self.hidden_layer_size, input_shape =\
                                    (None, None, self.input_dim), return_sequences = True)(masked)
        elif self.RNN_mode == 'GRU':
            RNN_out = GRU(self.hidden_layer_size, input_shape =\
                                        (None, None, self.input_dim), return_sequences = True)(masked)
        else:
            print('Sth wrong with the RNN_mode')
            
            
        dense_out = Dense(self.output_dim, input_shape = (None, None, self.hidden_layer_size), activation='sigmoid')(RNN_out)
        y_order = Input(batch_shape = (None, None, self.output_dim), name = 'y_order')
        merged = multiply([dense_out, y_order])

        def reduce_dim(x):
            x = K.max(x, axis = 2, keepdims = True)
            return x

        def reduce_dim_shape(input_shape):
            shape = list(input_shape)
            shape[-1] = 1
            print ("reduced_shape", shape)
            return tuple(shape)
        reduced = Lambda(reduce_dim, output_shape = reduce_dim_shape)(merged)
        model = Model(inputs=[x,y_order], outputs=reduced)
                
        
        earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')
        model.compile( optimizer = 'rmsprop',\
                       loss = self.custom_bce,\
                      metrics=['accuracy'])

        model.fit([self.train_index,self.train_order], self.train_response, batch_size = self.batch_size, \
                  epochs=self.epoch, \
                  callbacks = [ earlyStopping, \
                             TestCallback_y_order([self.val_index,self.val_order,\
                             self.val_response])],\
                             validation_data = ([self.val_index,self.val_order],self.val_response), shuffle = True)
        
        
        
        
        
        