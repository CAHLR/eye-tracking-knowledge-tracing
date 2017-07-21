# coding: utf-8
import pdb
import numpy as np
import pandas as pd
def inputStudent(csvInput):
    for line in csvInput:
        # nStep, questionsID, correct = yield(line)
        if line[-1]=='':
            yield line[:-1]
        else: yield line

def grid_eyetracking(w_size, h_size, x_min, y_min, x_max, y_max, position):
    #position should be a [x_list,y_list],
    position = np.array(list(zip(position[0],position[1])))
    x_axis = np.linspace(x_min, x_max, num = (w_size+1))
    y_axis = np.linspace(y_min, y_max, num = (h_size+1))
    index = -1 * np.ones(len(position)) #initialized to be -1
    for i in range(len(x_axis)-1):
        for j in range(len(y_axis)-1):
            
            pos_in_region = set(list(np.where(position[:,0]>=x_axis[i]))[0]) &\
            set(list(np.where(position[:,0]<=x_axis[i+1]))[0]) &\
            set(list(np.where(position[:,1]>=y_axis[j]))[0]) &\
            set(list(np.where(position[:,1]<=y_axis[j+1]))[0])
            index[list(pos_in_region)] = i + w_size*j  
    index[np.where(index == -1)] = w_size * h_size # new region for position out of range 
    return index

def transpose_df_column(df_input):
    #df_input is a pd.DataFrame data.
    #df_output is a pd.DataFrame data, which switches the values and columns to each other.
    new_columns = {}
    for ix, row in df_input.iterrows():
        for i in range(len(row)):
            if np.isnan(row[i]):
                continue
            if row[i] not in new_columns:
                new_columns.update({row[i]:[]})
            if len(new_columns[row[i]])>=1 and new_columns[row[i]][-1] == df_input.columns[i]:
                continue
            else:
                new_columns[row[i]].append(df_input.columns[i])
    df_output = pd.DataFrame.from_dict(new_columns, orient='index')
    df_output.transpose()
    df_output = df_output.sort_index()
    return df_output