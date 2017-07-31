# coding: utf-8
import numpy as np
import pandas as pd
import csv
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Merge
from theano import tensor as T
#from DKT import DKTnet
from keras.preprocessing import sequence
import pdb
import os
from utils import * #grid_eyetracking(w_size, h_size, x_min, y_min, x_max, y_max, position)
# This is the 30 students-5 cross validation version. 7.16
from eyetracking_model import eyetracking_net
#from optparse import OptionParser
import argparse
parser = argparse.ArgumentParser()





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
# This list is for Session 2, 20+5 students in total
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


answer_set = {'CORRECT':1, 'INCORRECT':0}
total_answers = []
total_name = []

standard = [] # standard is for choosing certain Screen or page. 

''' Starts processing data'''    
for imputed_file in train_val_data:

    print(imputed_file)
    
    student = pd.read_csv(imputed_file, low_memory=False)
    standard = (student.Stimulus == 'bl i.jpg')
    total_name.append(imputed_file)

    '''Now ignore unimportant values in questions'''
    student.loc[(student['Response']!='CORRECT') & (student['Response']!='INCORRECT'),'question'] = 'ignore'
    student.loc[(student.question =='_root') | (student.question == 'done'),'question'] = 'ignore'

    student.Response = student.Response.replace(float('nan'),-1)
    student.Response = student.Response.replace('HINT', -1)
    student.Response = student.Response.replace('CORRECT',1)
    student.Response = student.Response.replace('INCORRECT',0)

    total_answers.append(student.Response[standard])

print('find ',len(total_answers),'student')

answer_dict = {}

for i in range(len(total_name)):
    answer_dict.update({total_name[i]: total_answers[i]})
    
print('Data processing finished, now starts cross validation')
'''Now starts cross validation'''    
cv_count = 0
avg_acc = []
for indexes in cross_val_list:
    cv_count += 1
    print ('cv_count = ',cv_count)
    train_indexes, val_indexes = indexes
    acc = []
    for val_index in val_indexes:
        ans = answer_dict[train_val_data[val_index]]
        #acc.append(len(ans[ans==1])/(len(ans[ans==-1])))
        acc.append(len(ans[ans==1])/(len(ans[ans==1])+len(ans[ans==0])))
    acc = sum(acc)/len(acc)
    avg_acc.append(acc)
print('avg_acc',avg_acc)



























