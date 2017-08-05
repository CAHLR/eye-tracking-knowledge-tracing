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

parser.add_argument('--y_order', type=bool, default= False)
parser.add_argument('--file_name', type=str, default= 'output_file')

parser.add_argument('--simple_index', type=bool, default= False)
parser.add_argument('--simple_onehot', type=bool, default= False)
parser.add_argument('--simple_continuous', type=bool, default= False)
parser.add_argument('--complex_onehot', type=bool, default= False)
parser.add_argument('--complex_continuous', type=bool, default= False)

parser.add_argument('--batch_size', type= int, default= 9)
parser.add_argument('--hidden_layer_size', type= int, default= 256)
parser.add_argument('--learning_rate', type= float, default= 0.001)
parser.add_argument('--optimizer_mode', type= str, default= 'RMSprop')
parser.add_argument('--RNN_mode', type=str, default= 'SimpleRNN')

args = parser.parse_args()

batch_size = args.batch_size 
hidden_layer_size = args.hidden_layer_size
learning_rate = args.learning_rate
optimizer_mode = args.optimizer_mode
RNN_mode = args.RNN_mode



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

root = '/research/atoms/Session2/'
for dirpath,dirnames,filenames in os.walk(root):
    for filename in filenames:
        for i in range(len(train_val_data)):
            if filename.endswith(train_val_data[i]):
                train_val_data[i] = dirpath + '/' + filename
'''                

total_problems = []
total_answers = []
problem_ID = 0 # make each question to be an ID
problem_set = {} # unique problem type
answer_set = {'CORRECT':1, 'INCORRECT':0}

total_position = []
total_answers = []
total_name = []
total_index = []
total_time_stamp = []
total_page_num = []
total_SRquestion = []

'''For y_order'''
SRquestion_set = set()
SRquestion_ID = 0
SRquestion_dict = {}


x_min = float("inf")
y_min = float("inf")
x_max = 0 # initialization
y_max = 0 # initialization
w_size = 5
h_size = 5
epoch = 50

standard = [] # standard is for choosing certain Screen or page. 

''' Starts processing data'''    
for imputed_file in train_val_data:
    #if imputed_file[-19] == '/':
        #student_name = imputed_file[-18:]
    #else:
        #student_name = imputed_file[-19:]
    print(imputed_file)
    
    student = pd.read_csv(imputed_file, low_memory=False)
    standard = (student.Stimulus == 'bl i.jpg')
    #standard = (student.Screen_Number<=3)
    total_name.append(imputed_file)

    total_page_num.append(list(student.Representation_Number_Overall[standard]))

    total_position.append([student.L_Raw_X[standard],\
                           student.L_Raw_Y[standard]])

    '''Now ignore unimportant values in questions'''
    student.loc[(student['Response']!='CORRECT') & (student['Response']!='INCORRECT'),'question'] = 'ignore'
    student.loc[(student.question =='_root') | (student.question == 'done')|(student['question'] == 'nl'),'question'] = 'ignore'

    '''Create SRquestion'''

    '''Change type of numbers for creating SRquestion'''
    #student.Screen_Number = student.Screen_Number.astype(str)
    student.Representation_Number_Overall = student.Representation_Number_Overall.astype(str)

    '''Create column named SRquestion_name'''
    #student['SRquestion_name'] = student.Screen_Number + \
                    #student.Representation_Number_Within_Screen +\
                    #student.question
    student['SRquestion_name'] = student.Representation_Number_Overall + student.question

    '''Change type of numbers back'''
    #student.Screen_Number = student.Screen_Number.astype(float)
    student.Representation_Number_Overall = student.Representation_Number_Overall.astype(float)     

    '''Update the SRquestion set and dict'''
    # We only want SRquestion_set within certain Screen
    SRquestion_set.update(student.loc[(student.question!= 'ignore') &\
                              (standard),'SRquestion_name']) 

    # SRquestion_ID starts from 1. SRquestion_ID == 0 means padding.
    for SRquestion_name in SRquestion_set: 
        if SRquestion_name not in SRquestion_dict:
            SRquestion_ID += 1
            SRquestion_dict.update({SRquestion_name: SRquestion_ID})
    student['SRquestion_ID'] = 0 #initialize new column.

    tmp_condition = (student.question!='ignore')&(standard)
    student.loc[tmp_condition,'SRquestion_ID'] = student.loc[tmp_condition,'SRquestion_name'].\
                                                 apply(lambda x:SRquestion_dict[x])    
    tmp_condition = []

    total_SRquestion.append(student['SRquestion_ID'][standard])
    
    # We found the bug!
    student.loc[(student.question == 'ignore'),'Response'] = -1
    student.Response = student.Response.replace(float('nan'),-1)
    student.Response = student.Response.replace('HINT', -1)
    student.Response = student.Response.replace('CORRECT',1)
    student.Response = student.Response.replace('INCORRECT',0)

    total_answers.append(student.Response[standard])
    total_time_stamp.append(student[standard].iloc[:,0])

    '''To avoid(0,0) calculated as the min value'''
    student.L_Raw_X = student.L_Raw_X.replace(0.0,float('inf'))
    student.L_Raw_Y = student.L_Raw_Y.replace(0.0,float('inf'))
    x_min = min(x_min,min(student.L_Raw_X[standard]))
    y_min = min(y_min,min(student.L_Raw_Y[standard]))
    student.L_Raw_X = student.L_Raw_X.replace(float('inf'),0.0)
    student.L_Raw_Y = student.L_Raw_Y.replace(float('inf'),0.0)

    x_max = max(x_max,max(student.L_Raw_X[standard]))
    y_max = max(y_max,max(student.L_Raw_Y[standard]))
print('x_max is ',x_max)
print('y_max is ',y_max)
assert len(total_position) == len(total_answers)
print('find ',len(total_position),'student')

# Clear set and dict
SRquestion_set = set()
#SRquestion_ID = 0
SRquestion_dict = {} 



df = pd.DataFrame([], columns=[])
for i in range(len(total_position)):
    print(i)
    index = grid_eyetracking(w_size, h_size, x_min, y_min, x_max, y_max, total_position[i])
    df_new = pd.DataFrame(index, columns = [total_name[i]]) 
    df = pd.concat([df,df_new], axis=1)
df = df.replace(float('nan'),-1.0)

total_index = df.transpose().values.tolist()

'''Use unique and reduced page_num'''
'''We move it here from complex_onehot'''
page_reduce_dict = {}
page_reduce_ID = 0
total_tmp_page_num = []
for i in total_page_num:
    tmp_page_num = []
    for j in i:
        if j not in page_reduce_dict:
            page_reduce_ID += 1
            page_reduce_dict.update({j:page_reduce_ID})
        tmp_page_num.append(page_reduce_dict[j])
    total_tmp_page_num.append(tmp_page_num)
total_page_num = total_tmp_page_num
max_page = page_reduce_ID # The largest new page_num
total_tmp_page_num = []
tmp_page_num = []


        
index_dict = {}
answer_dict = {}
position_dict = {}
page_num_dict = {}
order_dict = {}
for i in range(len(total_name)):
    index_dict.update({total_name[i]: total_index[i]})
    position_dict.update({total_name[i]: total_position[i]})
    answer_dict.update({total_name[i]: total_answers[i]})
    page_num_dict.update({total_name[i]: total_page_num[i]})
    order_dict.update({total_name[i]: total_SRquestion[i]})
    
print('Data processing finished, now starts cross validation')
'''Now starts cross validation'''    
cv_count = 0
max_len = np.max([len(i) for i in total_index])
for indexes in cross_val_list:
    cv_count += 1
    print ('cv_count = ',cv_count)
    train_indexes, val_indexes = indexes
    train_input = []
    train_truth = []
    val_input = []
    val_truth = []
    
    
    '''Creating y_order matrix for all modes'''
    train_order = []
    val_order = []
    for train_index in train_indexes:
        pad_len = max_len-len(answer_dict[train_val_data[train_index]])    
        pad_order = list(order_dict[train_val_data[train_index]]) +\
                   list((-1.) * np.ones(pad_len))
        train_order.append(pad_order)
                    
    for val_index in val_indexes:
        pad_len = max_len-len(answer_dict[train_val_data[val_index]])           
        pad_order = list(order_dict[train_val_data[val_index]]) +\
                   list((-1.) * np.ones(pad_len))               
        val_order.append(pad_order)
        
    '''Create onehot for orders while zero-hot for no response time slice'''
    train_order = np.eye(SRquestion_ID+1)[np.array(train_order, int)] #SRquestion starts from 1,while 0 is the padding value
    val_order = np.eye(SRquestion_ID+1)[np.array(val_order, int)]        
    train_order = train_order[:,:,1:]
    val_order = val_order[:,:,1:]
        
        
        
    if args.simple_continuous == True:
        for train_index in train_indexes:
            pad_len = max_len-len(answer_dict[train_val_data[train_index]])
            pad_truth = list(answer_dict[train_val_data[train_index]]) +\
                   list((-1.) * np.ones(pad_len))
            pad_x = list(position_dict[train_val_data[train_index]][0]) +\
                   list((-1.) * np.ones(pad_len))
            pad_y = list(position_dict[train_val_data[train_index]][1]) +\
                   list((-1.) * np.ones(pad_len))
            
                
            train_input.append([pad_x, pad_y])
            train_truth.append(pad_truth)
            
        for val_index in val_indexes:
            pad_len = max_len-len(answer_dict[train_val_data[val_index]])
            pad_truth = list(answer_dict[train_val_data[val_index]]) +\
                   list((-1.) * np.ones(pad_len))
            pad_x = list(position_dict[train_val_data[val_index]][0]) +\
                   list((-1.) * np.ones(pad_len))
            pad_y = list(position_dict[train_val_data[val_index]][1]) +\
                   list((-1.) * np.ones(pad_len))
                
            pad_order = list(order_dict[train_val_data[val_index]]) +\
                   list((-1.) * np.ones(pad_len))               
            val_input.append([pad_x, pad_y])
            val_truth.append(pad_truth)
            
            
        train_input = (np.array(train_input)).swapaxes(1,2)
        train_truth = (np.array(train_truth))[:,:,np.newaxis]
        val_input = (np.array(val_input)).swapaxes(1,2)
        val_truth = (np.array(val_truth))[:,:,np.newaxis]
        
            
    if args.simple_index == True:
        for train_index in train_indexes:
            pad_len = max_len-len(answer_dict[train_val_data[train_index]])
            pad = list(answer_dict[train_val_data[train_index]]) +\
                   list((-1.) * np.ones(pad_len))
            train_input.append(index_dict[train_val_data[train_index]])
            train_truth.append(pad)
        for val_index in val_indexes:
            pad_len = max_len-len(answer_dict[train_val_data[val_index]])
            pad = list(answer_dict[train_val_data[val_index]]) +\
                   list((-1.) * np.ones(pad_len))                        
            val_input.append(index_dict[train_val_data[val_index]])
            val_truth.append(pad)
            
            #if len(np.array(train_input).shape) == 2: # input is region by integer, which is 2-d
        train_input = (np.array(train_input))[:,:,np.newaxis]
        train_truth = (np.array(train_truth))[:,:,np.newaxis]
        val_input = (np.array(val_input))[:,:,np.newaxis]
        val_truth = (np.array(val_truth))[:,:,np.newaxis]
    
    if args.simple_onehot == True:
        max_index = 0
        for train_index in train_indexes:
            max_index = max(max_index,train_index)
            pad_len = max_len-len(answer_dict[train_val_data[train_index]])
            pad = list(answer_dict[train_val_data[train_index]]) +\
                   list((-1.) * np.ones(pad_len))
            train_input.append(index_dict[train_val_data[train_index]])
            train_truth.append(pad)
        for val_index in val_indexes:
            max_index = max(max_index,val_index)
            pad_len = max_len-len(answer_dict[train_val_data[val_index]])
            pad = list(answer_dict[train_val_data[val_index]]) +\
                   list((-1.) * np.ones(pad_len))                        
            val_input.append(index_dict[train_val_data[val_index]])
            val_truth.append(pad)
        
        train_input = np.array(train_input, dtype = int)
        val_input = np.array(val_input, dtype = int)
        train_input[train_input==-1.] = 0 # padding value is 0
        val_input[val_input==-1.] = 0
        
        train_input = np.eye(max_index+2)[np.array(train_input)] # onehot
        train_truth = (np.array(train_truth))[:,:,np.newaxis]
        val_input = np.eye(max_index+2)[np.array(val_input)] #onehot
        val_truth = (np.array(val_truth))[:,:,np.newaxis]
        '''One hot will be of length( max_index+2)'''
        print('Max_index-',max_index,'Padding-',max_index+1,' Onehot-',max_index+2)
        
    
    if args.complex_continuous == True:
        train_page_num = []
        val_page_num = []
        total_max_page = []
        for i in total_page_num:
            total_max_page.append(max(i))
        max_page = max(total_max_page)

        for train_index in train_indexes:
            pad_len = max_len-len(answer_dict[train_val_data[train_index]])
            pad_truth = list(answer_dict[train_val_data[train_index]]) +\
                   list((-1.) * np.ones(pad_len))
            pad_x = list(position_dict[train_val_data[train_index]][0]) +\
                   list((-1.) * np.ones(pad_len))
            pad_y = list(position_dict[train_val_data[train_index]][1]) +\
                   list((-1.) * np.ones(pad_len))
            
            pad_page_num = list(page_num_dict[train_val_data[train_index]])+\
                            list((-1.) * np.ones(pad_len))
            train_input.append([pad_x, pad_y])
            train_truth.append(pad_truth)
            train_page_num.append(pad_page_num)
            
            
        for val_index in val_indexes:
            pad_len = max_len-len(answer_dict[train_val_data[val_index]])
            pad_truth = list(answer_dict[train_val_data[val_index]]) +\
                   list((-1.) * np.ones(pad_len))
            pad_x = list(position_dict[train_val_data[val_index]][0]) +\
                   list((-1.) * np.ones(pad_len))
            pad_y = list(position_dict[train_val_data[val_index]][1]) +\
                   list((-1.) * np.ones(pad_len))
            
            pad_page_num = list(page_num_dict[train_val_data[val_index]])+\
                        list((-1.) * np.ones(pad_len))
            
            val_input.append([pad_x, pad_y])
            val_truth.append(pad_truth)
            val_page_num.append(pad_page_num)
            
        #train_page_num = np.array
        
        train_input = (np.array(train_input)).swapaxes(1,2)
        train_truth = (np.array(train_truth))[:,:,np.newaxis]
        val_input = (np.array(val_input)).swapaxes(1,2)
        val_truth = (np.array(val_truth))[:,:,np.newaxis]
        train_page_num = np.eye(int(max_page))[np.array(train_page_num,int)-1] # must be int for onehot
        val_page_num = np.eye(int(max_page))[np.array(val_page_num,int)-1] # -1 because the ID should starts from 0, for onehot
        train_input = np.concatenate((train_input,train_page_num),axis = 2)
        val_input = np.concatenate((val_input,val_page_num),axis = 2)
        
        
        
    if args.complex_onehot == True:
        train_page_num = []
        val_page_num = []

        max_index = 0 
        
        for train_index in train_indexes:
            max_index = max(max_index,train_index)
            pad_len = max_len-len(answer_dict[train_val_data[train_index]])
            pad = list(answer_dict[train_val_data[train_index]]) +\
                   list((-1.) * np.ones(pad_len))
            pad_page_num = list(page_num_dict[train_val_data[train_index]])+\
                            list((-1.) * np.ones(pad_len))    
            train_input.append(index_dict[train_val_data[train_index]])
            train_truth.append(pad)
            train_page_num.append(pad_page_num)
            
        for val_index in val_indexes:
            max_index = max(max_index,val_index)
            pad_len = max_len-len(answer_dict[train_val_data[val_index]])
            pad = list(answer_dict[train_val_data[val_index]]) +\
                   list((-1.) * np.ones(pad_len))  
            pad_page_num = list(page_num_dict[train_val_data[val_index]])+\
                        list((-1.) * np.ones(pad_len))                
            val_input.append(index_dict[train_val_data[val_index]])
            val_truth.append(pad)
            val_page_num.append(pad_page_num)
        
        train_input = np.array(train_input, dtype = int)
        val_input = np.array(val_input, dtype = int)
        train_input[train_input==-1.] = 0 # padding value is 0
        val_input[val_input==-1.] = 0
        
        train_input = np.array(train_input)
        val_input = np.array(val_input)
        train_page_num = np.array(train_page_num)
        val_page_num = np.array(val_page_num)
        
        # page num starts from 1
        # notice that we minus 1 when calculate the new train_input. If the question is on the 1st page, then it's '0 + original ID'
        train_input[np.where(train_input>0)] = (train_page_num[np.where(train_input>0)]-1)*\
                                                (max_index+1) + train_input[np.where(train_input>0)]
        val_input[np.where(val_input>0)] = (val_page_num[np.where(val_input>0)]-1)*\
                                                (max_index+1)+ val_input[np.where(val_input>0)]
        
        
        train_input = np.eye(int(1+max_page*(max_index+1)))[np.array(train_input,dtype = int)] # onehot
        train_truth = (np.array(train_truth))[:,:,np.newaxis]
        val_input = np.eye(int(1+max_page*(max_index+1)))[np.array(val_input,dtype = int)] #onehot
        val_truth = (np.array(val_truth))[:,:,np.newaxis]
        
        
        '''One hot will be of length( max_index+2)'''
        print('Max_index-',max_index,'Padding-',max_index+1,' Onehot-',max_index+2)
        
    

    if args.y_order == True:
        print('y_order==True')
        input_dim = train_input.shape[-1]
        output_dim = train_order.shape[-1]
        model = eyetracking_net(batch_size, epoch, hidden_layer_size, input_dim, output_dim,\
                           learning_rate, optimizer_mode, RNN_mode)
        model.build_y_order(train_input, train_order, train_truth,  val_input, val_order, val_truth, args.file_name)
    
    elif args.y_order == False:
        print('y_order==False')
        input_dim = train_input.shape[-1]
        output_dim = 1
        model = eyetracking_net(batch_size, epoch, hidden_layer_size, input_dim, output_dim,\
                           learning_rate, optimizer_mode, RNN_mode)
        model.build_no_y_order(train_input,train_truth, val_input,val_truth)
    else:
        print('Sth wrong about y_order when applying to model')

        



