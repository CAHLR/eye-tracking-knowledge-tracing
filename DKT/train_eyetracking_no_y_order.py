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
from optparse import OptionParser

'''Stop updating since 7/27/2017, see further development in train_eyetracking.py'''

'''Choose different mode of eye-tracking model'''
parser = OptionParser()
parser.add_option("--simple_index",action="store_true",  dest="simple_index",\
                 help="using simple index for training, no onehot,and difference between pages")
parser.add_option("--simple_onehot",action="store_true", dest="simple_onehot",\
                 help="using one-hot for training, no difference between pages")
parser.add_option("--simple_continuous",action="store_true",  dest="simple_continuous",\
                 help="using simple continuous for training, no difference between pages")

parser.add_option("--complex_onehot",action="store_true",  dest="complex_onehot",\
                 help="using complex onehot for training, have difference between pages")
parser.add_option("--complex_continuous",action="store_true",  dest="complex_continuous",\
                 help="using complex continuous for training, have difference between pages")

(options, args) = parser.parse_args()





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




root = '/research/atoms/Session2'
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
total_page_num = []

imputed_file = imputed_list[0]
x_min = float("inf")
y_min = float("inf")
x_max = 0 # initialization
y_max = 0 # initialization
w_size = 5
h_size = 5
batch_size = 5
epoch = 50
hidden_layer_size = 128
screen_num = 3
''' Starts processing data'''    
for imputed_file in imputed_list:
    if imputed_file[-19] == '/':
        student_name = imputed_file[-18:]
    else:
        student_name = imputed_file[-19:]
    if student_name in train_val_data:
        print(student_name)
        student = pd.read_csv(imputed_file, low_memory=False)
        total_name.append(student_name)
        
        total_page_num.append(list(student.Representation_Number_Overall[student.Screen_Number<=screen_num]))
        
        total_position.append([student.L_Raw_X[student.Screen_Number<=screen_num],\
                               student.L_Raw_Y[student.Screen_Number<=screen_num]])
        student.Response = student.Response.replace(float('nan'),-1)
        student.Response = student.Response.replace('HINT', -1)
        student.Response = student.Response.replace('CORRECT',1)
        student.Response = student.Response.replace('INCORRECT',0)
        total_answers.append(student.Response[student.Screen_Number<= screen_num])
        total_time_stamp.append(student[student.Screen_Number<= screen_num].iloc[:,0])
        #to avoid(0,0) calculated as the min value
        student.L_Raw_X = student.L_Raw_X.replace(0.0,float('inf'))
        student.L_Raw_Y = student.L_Raw_Y.replace(0.0,float('inf'))
        x_min = min(x_min,min(student.L_Raw_X[student.Screen_Number<= screen_num]))
        y_min = min(y_min,min(student.L_Raw_Y[student.Screen_Number<= screen_num]))
        student.L_Raw_X = student.L_Raw_X.replace(float('inf'),0.0)
        student.L_Raw_Y = student.L_Raw_Y.replace(float('inf'),0.0)
        
        x_max = max(x_max,max(student.L_Raw_X[student.Screen_Number<= screen_num]))
        y_max = max(y_max,max(student.L_Raw_Y[student.Screen_Number<= screen_num]))
print('x_max is ',x_max)
print('y_max is ',y_max)
assert len(total_position) == len(total_answers)
print('find ',len(total_position),'student')


df = pd.DataFrame([], columns=[])
for i in range(len(total_position)):
    print(i)
    index = grid_eyetracking(w_size, h_size, x_min, y_min, x_max, y_max, total_position[i])
    df_new = pd.DataFrame(index, columns = [total_name[i]]) 
    df = pd.concat([df,df_new], axis=1)
df = df.replace(float('nan'),-1.0)

total_index = df.transpose().values.tolist()
index_dict = {}
answer_dict = {}
position_dict = {}
page_num_dict = {}
for i in range(len(total_name)):
    index_dict.update({total_name[i]: total_index[i]})
    position_dict.update({total_name[i]: total_position[i]})
    answer_dict.update({total_name[i]: total_answers[i]})
    page_num_dict.update({total_name[i]: total_page_num[i]})
    
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
    
    if options.simple_continuous == True:
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
                           
            val_input.append([pad_x, pad_y])
            val_truth.append(pad_truth)
            
            
        train_input = (np.array(train_input)).swapaxes(1,2)
        train_truth = (np.array(train_truth))[:,:,np.newaxis]
        val_input = (np.array(val_input)).swapaxes(1,2)
        val_truth = (np.array(val_truth))[:,:,np.newaxis]
        
            
    if options.simple_index == True:
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
    
    if options.simple_onehot == True:
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
        train_input[train_input==-1.] = 0 # padding value is max_index+1
        val_input[val_input==-1.] = 0
        
        train_input = np.eye(max_index+2)[np.array(train_input)] # onehot
        train_truth = (np.array(train_truth))[:,:,np.newaxis]
        val_input = np.eye(max_index+2)[np.array(val_input)] #onehot
        val_truth = (np.array(val_truth))[:,:,np.newaxis]
        '''One hot will be of length( max_index+2)'''
        print('Max_index-',max_index,'Padding-',max_index+1,' Onehot-',max_index+2)
        
    
    if options.complex_continuous == True:
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
        
        
        
    if options.complex_onehot == True:
        train_page_num = []
        val_page_num = []
        total_max_page = []
        for i in total_page_num:
            total_max_page.append(max(i))
        max_page = max(total_max_page)
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
        train_input[train_input==-1.] = 0 # padding value is max_index+1
        val_input[val_input==-1.] = 0
        
        train_input = np.array(train_input)
        val_input = np.array(val_input)
        train_page_num = np.array(train_page_num)
        val_page_num = np.array(val_page_num)
        
        # page num starts from 1
        train_input[np.where(train_input>0)] = train_page_num[np.where(train_input>0)]*\
                                                (max_index+1) + train_input[np.where(train_input>0)]
        val_input[np.where(val_input>0)] = val_page_num[np.where(val_input>0)]*\
                                                (max_index+1)+ val_input[np.where(val_input>0)]
        
        
        train_input = np.eye(int(1+max_page*(max_index+1)))[np.array(train_input,dtype = int)] # onehot
        train_truth = (np.array(train_truth))[:,:,np.newaxis]
        val_input = np.eye(int(1+max_page(max_index+1)))[np.array(val_input,dtype = int)] #onehot
        val_truth = (np.array(val_truth))[:,:,np.newaxis]
        
        
        '''One hot will be of length( max_index+2)'''
        print('Max_index-',max_index,'Padding-',max_index+1,' Onehot-',max_index+2)
        
        
    
    input_dim = train_input.shape[-1]
    output_dim = 1 # Ignore y_order information and only predict Correctness.
    model = eyetracking_net(batch_size, epoch, hidden_layer_size, input_dim, output_dim,\
                                learning_rate, optimizer_mode, RNN_mode)
    model.build_no_y_order(train_input,train_truth, val_input,val_truth)




        
