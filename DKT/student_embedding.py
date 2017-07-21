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
#from model import DKTnet
from DKT import DKTnet
from keras.preprocessing import sequence
import pdb
import os
from utils import * #grid_eyetracking(w_size, h_size, x_min, y_min, x_max, y_max, position)
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




root = '/research/atoms/'
imputed_list = []
for dirpath,dirnames,filenames in os.walk(root):
    for filename in filenames:
        if filename.endswith('_imputed.csv'):
            imputed_list.append(dirpath + '/' + filename)

total_problems = []
total_answers = []
problem_ID = 0 # make each question to be an ID
problem_set = {} # unique problem type
answer_set = {'CORRECT':1, 'INCORRECT':0}
print ('The num of students is ', len(imputed_list))

total_position = []
total_answers = []
total_name = []
total_index = []
total_time_stamp = []

imputed_file = imputed_list[0]
x_min = float("inf")
y_min = float("inf")
x_max = 0 # initialization
y_max = 0 # initialization
w_size = 5
h_size = 5
batch_size = 5
epoch = 10
hidden_layer_size = 128
    
for imputed_file in imputed_list:
    if imputed_file[-19] == '/':
        student_name = imputed_file[-18:]
    else:
        student_name = imputed_file[-19:]
    if student_name in train_val_data:
        print(student_name)
        student = pd.read_csv(imputed_file, low_memory=False)
        total_name.append(student_name)
        total_position.append([student.L_Raw_X[student.Screen_Number==1],student.L_Raw_Y[student.Screen_Number==1]])
        total_answers.append(student.Response[student.Screen_Number==1])
        total_time_stamp.append(student[student.Screen_Number==1].iloc[:,0])
        #to avoid(0,0) calculated as the min value
        student.L_Raw_X = student.L_Raw_X.replace(0.0,float('inf'))
        student.L_Raw_Y = student.L_Raw_Y.replace(0.0,float('inf'))
        x_min = min(x_min,min(student.L_Raw_X[student.Screen_Number==1]))
        y_min = min(y_min,min(student.L_Raw_Y[student.Screen_Number==1]))
        student.L_Raw_X = student.L_Raw_X.replace(float('inf'),0.0)
        student.L_Raw_Y = student.L_Raw_Y.replace(float('inf'),0.0)
        
        x_max = max(x_max,max(student.L_Raw_X[student.Screen_Number==1]))
        y_max = max(y_max,max(student.L_Raw_Y[student.Screen_Number==1]))
print('x_max is ',x_max)
print('y_max is ',y_max)
assert len(total_position) == len(total_answers)
print('find ',len(total_position),'student')

for position in total_position:
    index = grid_eyetracking(w_size, h_size, x_min, y_min, x_max, y_max, position)
    total_index.append(list(index))
    
df = pd.DataFrame([], columns=[])
for i in range(len(total_position)):
    print(i)
    index = grid_eyetracking(w_size, h_size, x_min, y_min, x_max, y_max, total_position[i])
    df_new = pd.DataFrame(index, columns = [total_name[i]]) 
    df = pd.concat([df,df_new], axis=1)
new_df = transpose_df_column(df)
print('transpose_df_column finished!')
#import pdb
#import matplotlib.pyplot as plt
#fig = plt.figure(figsize=(15,15))
#count = -1
#for i in range(len(total_position)):
    #plt.scatter(total_position[i][0][:count],total_position[i][1][:count])
    #plt.xlabel("x_label")
    #plt.ylabel("y_label")
    #plt.title("eye tracks in eyetracking")

#plt.xlim(x_min,x_max)
#plt.ylim(y_min,y_max)
#plt.legend()
#plt.show()