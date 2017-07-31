# coding: utf-8
import numpy as np
import csv
import utils
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Merge
from theano import tensor as T
from data_response import DataMatrix, student
#from model import DKTnet
from DKT import DKTnet
from keras.preprocessing import sequence
import pdb



# This list is for Session 1, 36+9 students in total
cross_val_list = [
    [[0, 1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 40, 41, 42, 44], 
     [2, 6, 13, 14, 19, 23, 38, 39, 43]], 
    [[0, 1, 2, 3, 5, 6, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 41, 43], 
     [4, 7, 9, 16, 26, 33, 40, 42, 44]], 
    [[0, 1, 2, 3, 4, 6, 7, 9, 12, 13, 14, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44], 
     [5, 8, 10, 11, 15, 18, 20, 24, 31]], 
    [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 28, 31, 33, 34, 35, 38, 39, 40, 42, 43, 44], 
     [0, 22, 27, 29, 30, 32, 36, 37, 41]], 
    [[0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 22, 23, 24, 26, 27, 29, 30, 31, 32, 33, 36, 37, 38, 39, 40, 41, 42, 43, 44], 
     [1, 3, 12, 17, 21, 25, 28, 34, 35]]]

train_val_data = [
 '/research/atoms/Session1/2014 F Session 1/atoms38_imputed.csv',
 '/research/atoms/Session1/2014 F Session 1/atoms26_imputed.csv',
 '/research/atoms/Session1/2014 F Session 1/atoms14_imputed.csv',
 '/research/atoms/Session1/2014 F Session 1/atoms49_imputed.csv',
 '/research/atoms/Session1/2014 F Session 1/atoms18_imputed.csv',
 '/research/atoms/Session1/2014 F Session 1/atoms28_imputed.csv',
 '/research/atoms/Session1/2014 F Session 1/atoms23_imputed.csv',
 '/research/atoms/Session1/2014 F Session 1/atoms9_imputed.csv',
 '/research/atoms/Session1/2014 F Session 1/atoms33_imputed.csv',
 '/research/atoms/Session1/2014 F Session 1/atoms4_imputed.csv',
 '/research/atoms/Session1/2014 F Session 1/atoms10_imputed.csv',
 '/research/atoms/Session1/2014 I Session 1/atoms7_imputed.csv',
 '/research/atoms/Session1/2014 I Session 1/atoms27_imputed.csv',
 '/research/atoms/Session1/2014 I Session 1/atoms3_imputed.csv',
 '/research/atoms/Session1/2014 I Session 1/atoms13_imputed.csv',
 '/research/atoms/Session1/2014 I Session 1/atoms5_imputed.csv',
 '/research/atoms/Session1/2014 I Session 1/atoms35_imputed.csv',
 '/research/atoms/Session1/2014 I Session 1/atoms46_imputed.csv',
 '/research/atoms/Session1/2014 I Session 1/atoms11_imputed.csv',
 '/research/atoms/Session1/2014 I Session 1/atoms29_imputed.csv',
 '/research/atoms/Session1/2014 I Session 1/atoms30_imputed.csv',
 '/research/atoms/Session1/2014 I Session 1/atoms2_imputed.csv',
 '/research/atoms/Session1/2014 S Session 1/atoms31_imputed.csv',
 '/research/atoms/Session1/2014 S Session 1/atoms47_imputed.csv',
 '/research/atoms/Session1/2014 S Session 1/atoms44_imputed.csv',
 '/research/atoms/Session1/2014 S Session 1/atoms15_imputed.csv',
 '/research/atoms/Session1/2014 S Session 1/atoms19_imputed.csv',
 '/research/atoms/Session1/2014 S Session 1/atoms37_imputed.csv',
 '/research/atoms/Session1/2014 S Session 1/atoms50_imputed.csv',
 '/research/atoms/Session1/2014 S Session 1/atoms6_imputed.csv',
 '/research/atoms/Session1/2014 S Session 1/atoms12_imputed.csv',
 '/research/atoms/Session1/2014 S Session 1/atoms36_imputed.csv',
 '/research/atoms/Session1/2014 S Session 1/atoms43_imputed.csv',
 '/research/atoms/Session1/2014 S Session 1/atoms41_imputed.csv',
 '/research/atoms/Session1/2014 SF Session 1/atoms22_imputed.csv',
 '/research/atoms/Session1/2014 SF Session 1/atoms42_imputed.csv',
 '/research/atoms/Session1/2014 SF Session 1/atoms17_imputed.csv',
 '/research/atoms/Session1/2014 SF Session 1/atoms39_imputed.csv',
 '/research/atoms/Session1/2014 SF Session 1/atoms8_imputed.csv',
 '/research/atoms/Session1/2014 SF Session 1/atoms45_imputed.csv',
 '/research/atoms/Session1/2014 SF Session 1/atoms21_imputed.csv',
 '/research/atoms/Session1/2014 SF Session 1/atoms20_imputed.csv',
 '/research/atoms/Session1/2014 SF Session 1/atoms1_imputed.csv',
 '/research/atoms/Session1/2014 SF Session 1/atoms32_imputed.csv',
 '/research/atoms/Session1/2014 SF Session 1/atoms34_imputed.csv']

'''
# This is the 30 students-5 cross validation version. 7.16
cross_val_list = [[[ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
       22, 23, 24],[0, 1, 2, 3, 4]],
[[ 0,  1,  2,  3,  4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
       22, 23, 24],[5, 6, 7, 8, 9]],
[[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 15, 16, 17, 18, 19, 20, 21,
       22, 23, 24],[10, 11, 12, 13, 14]],
[[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 20, 21,
       22, 23, 24],[15, 16, 17, 18, 19]],
[[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19], [20, 21, 22, 23, 24]]]

train_val_data = ['atoms31_imputed.csv', 'atoms21_imputed.csv', 'atoms28_imputed.csv', 'atoms15_imputed.csv', 'atoms20_imputed.csv', 'atoms37_imputed.csv', 'atoms18_imputed.csv', 'atoms36_imputed.csv', 'atoms07_imputed.csv', 'atoms6_imputed.csv', 'atoms1_imputed.csv', 'atoms38_imputed.csv', 'atoms45_imputed.csv', 'atoms27_imputed.csv', 'atoms9_imputed.csv', 'atoms22_imputed.csv', 'atoms4_imputed.csv', 'atoms8_imputed.csv', 'atoms29_imputed.csv', 'atoms14_imputed.csv', 'atoms12_imputed.csv', 'atoms17_imputed.csv', 'atoms43_imputed.csv', 'atoms26_imputed.csv', 'atoms19_imputed.csv']

test_data = ['atoms11_imputed.csv', 'atoms32_imputed.csv', 'atoms41_imputed.csv', 'atoms33_imputed.csv', 'atoms39_imputed.csv', 'atoms10_imputed.csv', 'atoms23_imputed.csv']
'''
data = DataMatrix()
data.build()
print('DataMatrix initialized')
stu_count = 0
stu_dict = {}
for student in data.trainData:
    stu_dict.update({student.name[0]:stu_count})
    stu_count += 1
assert stu_count == len(data.trainData)
    
for i in range(len(train_val_data)):
    if train_val_data[i][-19] == '/':
        train_val_data[i] = train_val_data[i][-18:]
    else:
        train_val_data[i] = train_val_data[i][-19:]
    
    
cv_count = 0
for indexes in cross_val_list:
    cv_count += 1
    print ('cv_count = ',cv_count)
    train_indexes, val_indexes = indexes
    train_fold = []
    val_fold = []
    
    for train_index in train_indexes:
        train_fold.append(data.trainData[stu_dict[(train_val_data[train_index])]])
    for val_index in val_indexes:
        val_fold.append(data.trainData[stu_dict[(train_val_data[val_index])]])
    #batch_size = 5
    batch_size = 6
    input_dim_order =  int(data.max_questionID + 1) #consider whether we need plus 1
    input_dim = 2 * input_dim_order
    epoch = 10
    hidden_layer_size = 64

    '''Training part starts from now'''
    x_train = []
    y_train = []
    y_train_order = []
    num_student = 0
    for student in train_fold:
        num_student += 1
        if num_student % 200 ==0:
            print (num_student,' ',num_student/46051.)
        x_single_train = np.zeros([input_dim, data.longest])
        y_single_train = np.zeros([1, data.longest])
        y_single_train_order = np.zeros([input_dim_order, data.longest])

        for i in range(student.n_answers):
            if student.correct[i] == 1.: # if correct
                x_single_train[student.ID[i]*2-1, i] = 1.
            elif student.correct[i] == 0.: # if wrong
                x_single_train[student.ID[i]*2, i] = 1.
            else:
                print (student.correct[i])
                print ("wrong length with student's n_answers or correct")
            y_single_train[0, i] = student.correct[i]
            y_single_train_order[student.ID[i], i] = 1.
        for i in  range(data.longest-student.n_answers):
            x_single_train[:,student.n_answers + i] = -1
            y_single_train[:,student.n_answers + i] = -1
            #notice that the padding value of order is still zero.
            y_single_train_order[:,student.n_answers + i] = 0
        x_single_train = np.transpose(x_single_train)
        y_single_train = np.transpose(y_single_train)
        y_single_train_order = np.transpose(y_single_train_order)
        x_train.append(x_single_train)
        y_train.append(y_single_train)
        y_train_order.append(y_single_train_order)
    print ("train num students", num_student)
    #x_train = sequence.pad_sequences(x_train, maxlen=1000, dtype='float64',padding='post', truncating='post', value=-1.)
    print ('preprocessing finished')
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train_order = np.array(y_train_order)
    #now the dimensions are samples, time_length, questions
    #this procedure is for matching the pred answer of next question to the groundtruth of next question.
    x_train = x_train[:,:-1,:]
    y_train = y_train[:,1:,:]
    y_train_order = y_train_order[:,1:,:]




    '''validation part starts from now'''
    x_val = []
    y_val = []
    y_val_order = []
    num_student = 0
    for student in val_fold:
        num_student += 1
        if num_student % 200 ==0:
            print (num_student,' ',num_student/46051.)
        x_single_val = np.zeros([input_dim, data.longest])
        y_single_val = np.zeros([1, data.longest])
        y_single_val_order = np.zeros([input_dim_order, data.longest])

        for i in range(student.n_answers):
            if student.correct[i] == 1.: # if correct
                x_single_val[student.ID[i]*2-1, i] = 1.
            elif student.correct[i] == 0.: # if wrong
                x_single_val[student.ID[i]*2, i] = 1.
            else:
                print (student.correct[i])
                print ("wrong length with student's n_answers or correct")
            y_single_val[0, i] = student.correct[i]
            y_single_val_order[student.ID[i], i] = 1.
        for i in  range(data.longest-student.n_answers):
            x_single_val[:,student.n_answers + i] = -1
            y_single_val[:,student.n_answers + i] = -1
            #notice that the padding value of order is still zero.
            y_single_val_order[:,student.n_answers + i] = 0
        x_single_val = np.transpose(x_single_val)
        y_single_val = np.transpose(y_single_val)
        y_single_val_order = np.transpose(y_single_val_order)
        x_val.append(x_single_val)
        y_val.append(y_single_val)
        y_val_order.append(y_single_val_order)
    print ("val num students", num_student)
    #x_val = sequence.pad_sequences(x_val, maxlen=1000, dtype='float64',padding='post', truncating='post', value=-1.)
    print ('preprocessing finished')
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    y_val_order = np.array(y_val_order)
    #now the dimensions are samples, time_length, questions
    #this procedure is for matching the pred answer of next question to the groundtruth of next question.
    x_val = x_val[:,:-1,:]
    y_val = y_val[:,1:,:]
    y_val_order = y_val_order[:,1:,:]
    model = DKTnet(input_dim, input_dim_order, hidden_layer_size,
            batch_size, epoch, np.array(x_train), np.array(y_train), np.array(y_train_order),np.array(x_val), np.array(y_val), np.array(y_val_order))
    
    model.build()

