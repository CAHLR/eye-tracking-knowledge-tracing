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
import pandas as pd

"""
To calculate and print the average rmse on the validation set after every epoch
"""
class TestCallback(Callback):

    def __init__(self, test_data = [[],[],[]]):
        self.x_test, self.y_test_order, self.y_test = test_data
        self.count = 0

    def on_epoch_begin(self, epoch, logs={}):
        y_pred = self.model.predict([self.x_test, self.y_test_order])
        avg_rmse, avg_acc = self.rmse_masking(self.y_test, y_pred)
        self.output_info(self.y_test, y_pred)
        print('\nTesting avg_rmse: {}\n'.format(avg_rmse))
        print('\nTesting avg_acc: {}\n'.format(avg_acc))

    def output_info(self, y_true, y_pred):
        '''output_info'''
        time_seq_order = np.argmax(self.y_test_order, axis=2)
        
        time_seq_order = time_seq_order.flatten()
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        seq_order = time_seq_order[y_true!=-1]
        seq_true = y_true[y_true!=-1]
        seq_pred = y_pred[y_true!=-1]
        
        order_name = ('order'+str(self.count))
        true_name = ('true'+str(self.count))
        pred_name = ('pred'+str(self.count))
        df = pd.DataFrame({order_name:seq_order, true_name:seq_true,\
                          pred_name:seq_pred})
        #df.order_name = df.order_name.astype(str)
        #df.true_name = df.true_name.astype(str)
        #df.pred_name = df.pred_name.astype(str)
        self.count += 1
        print('output log')
        path = '/research/atoms/log/DKT_sanity_log.csv'
        df.to_csv(path,mode = 'a')
        

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
                #if mask_matrix[i] == 0:
                    ##break
                    #continue
                if y_true[i] ==-1:
                    continue
                if y_true[i] == 1 and y_pred[i] >0.5:
                    correct += 1
                elif y_true[i] == 0 and y_pred[i] < 0.5:
                    correct += 1
                #elif y_true[i] == -1:
                    #padding_num += 1
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
                    #break
                    continue
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

class DKTnet():

    def __init__(self, input_dim, input_dim_order, hidden_layer_size, batch_size, epoch,
        x_train=[], y_train=[], y_train_order=[],
        x_test=[], y_test=[], y_test_order=[]):
        ## input dim is the dimension of the input at one timestamp (dimension of x_t)
        self.input_dim = int(input_dim)
        ## input_dim_order is the dimension of the one hot representation of problem to
        ## check the order of occurence of responses according to timestamp
        self.input_dim_order = int(input_dim_order)

        self.hidden_layer_size = hidden_layer_size
        self.batch_size = int(batch_size)
        self.epoch = int(epoch)

        ## xtrain is a 3D matrix of size ( samples * number of timestamp * dimension of input vec (x_t) )
        ## in cognitive tutor # of students * # total responses * # input_dim
        self.x_train = x_train
        ## y_train is a matrix of ( samples * one hot representation according to problem output value at each timestamp (y_t) )
        self.y_train = y_train
        ## y_train_order is the one hot representation of problem according to timestamp starting from
        ## t=1 if training starts at t=0
        self.y_train_order = y_train_order

        self.x_test = x_test
        self.y_test = y_test
        self.y_test_order = y_test_order
        self.users = np.shape(x_train)[0]
        self.validation_split = 0.2
        print ("Initialization Done")

    def build_train_on_batch(self):

        ## first layer for the input (x_t)
        x = Input(batch_shape = (None, None, self.input_dim), name='x')
        masked = (Masking(mask_value= -1, input_shape = (None, None, self.input_dim)))(x)
        rnn_out = SimpleRNN(self.hidden_layer_size, input_shape = (None, None, self.input_dim), return_sequences = True)(masked)
        dense_out = Dense(self.input_dim_order, input_shape = (None, None, self.hidden_layer_size), activation='sigmoid')(rnn_out)
        y_order = Input(batch_shape = (None, None, self.input_dim_order), name = 'y_order')
        merged = multiply([dense_out, y_order])

        def reduce_dim(x):
            x = K.max(x, axis = 2, keepdims = True)
            return x

        def reduce_dim_shape(input_shape):
            shape = list(input_shape)
            shape[-1] = 1
            print ("reduced_shape", shape)
            return tuple(shape)

        earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')
        reduced = Lambda(reduce_dim, output_shape = reduce_dim_shape)(merged)
        self.model = Model(inputs=[x,y_order], outputs=reduced)
        self.model.compile( optimizer = 'rmsprop',
                        loss = 'binary_crossentropy',
                        metrics=['accuracy'])

    def train_on_batch(self, x_train, y_train, y_train_order):
        self.model.train_on_batch([x_train, y_train_order], y_train)

    def predict(self, x_val, y_val_order):
        y_pred = self.model.predict([x_val,y_val_order])
        return y_pred

    def build(self):
        ## first layer for the input (x_t)
        x = Input(batch_shape = (None, None, self.input_dim), name='x')
        masked = (Masking(mask_value= -1, input_shape = (None, None, self.input_dim)))(x)
        rnn_out = SimpleRNN(self.hidden_layer_size, input_shape = (None, None, self.input_dim), return_sequences = True)(masked)
        dense_out = Dense(self.input_dim_order, input_shape = (None, None, self.hidden_layer_size), activation='sigmoid')(rnn_out)
        y_order = Input(batch_shape = (None, None, self.input_dim_order), name = 'y_order')
        merged = multiply([dense_out, y_order])

        def reduce_dim(x):
            x = K.max(x, axis = 2, keepdims = True)
            return x

        def reduce_dim_shape(input_shape):
            shape = list(input_shape)
            shape[-1] = 1
            print ("reduced_shape", shape)
            return tuple(shape)
        #using 'loss' instead of 'val_loss'
        earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')
        reduced = Lambda(reduce_dim, output_shape = reduce_dim_shape)(merged)
        model = Model(inputs=[x,y_order], outputs=reduced)
        model.compile( optimizer = 'rmsprop',
                        loss = 'binary_crossentropy',
                        metrics=['accuracy'])



        #histories = my_callbacks.Histories()
        #print ("x_test",np.shape(self.x_test))
        #print ("y_test_order",np.shape(self.y_test_order))

        #layer_name = 'my_layer'
        #intermediate_layer_model = Model(inputs = [x,y_order],outputs = reduced)
        #intermediate_output = intermediate_layer_model.predict([self.x_train,self.y_train_order])
        #print ('reduced',intermediate_output.shape)
    
        model.fit([self.x_train, self.y_train_order], self.y_train, batch_size = self.batch_size, \
                  epochs=self.epoch, \
                  callbacks = [ earlyStopping, \
                                TestCallback((self.x_test, \
                                self.y_test_order, \
                                self.y_test))],\
            validation_data = ([self.x_test,self.y_test_order],self.y_test), shuffle = True)
        
        #model.fit([self.x_train, self.y_train_order], self.y_train, batch_size = self.batch_size, \
                  #epochs=self.epoch, \
                  #callbacks = [ earlyStopping, \
                                #TestCallback((self.x_train[int((1-self.validation_split)*self.users):], \
                                #self.y_train_order[ int((1-self.validation_split)*self.users):], \
                                #self.y_train[ int((1-self.validation_split)*self.users):]))], \
                  #validation_split = self.validation_split, shuffle = True)

