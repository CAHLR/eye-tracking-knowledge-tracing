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

data = DataMatrix()
data.build()
stu_count = 0
stu_dict = {}
for student in data.trainData:
    stu_dict.update({student.name[0]:stu_count})
    stu_count += 1
assert stu_count == len(data.trainData)
    
total_acc = []
for indexes in cross_val_list:
    train_indexes, val_indexes = indexes
    train_fold = []
    val_fold = []
    for train_index in train_indexes:
        train_fold.append(data.trainData[stu_dict[(train_val_data[train_index])]])
    for val_index in val_indexes:
        val_fold.append(data.trainData[stu_dict[(train_val_data[val_index])]])
    acc = []
    for student in val_fold:
        acc.append(sum(student.correct)/student.n_answers)
    avg_acc = sum(acc)/len(acc)
    total_acc.append(avg_acc)
print (total_acc)
       
        
    
    
    
    
    