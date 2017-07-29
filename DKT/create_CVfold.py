# coding: utf-8
import csv
import numpy as np
import pandas as pd
import pdb
import os
import pdb
import argparse
from sklearn.model_selection import KFold

'''This file is created for creating CV fold list of imputed students in eyetracking project.'''

parser = argparse.ArgumentParser()

parser.add_argument('--Session1', type=bool, default = False)
parser.add_argument('--Session2',type=bool, default = False)

args = parser.parse_args()
if args.Session1 == True:
    root = '/research/atoms/Session1/'
elif args.Session2 == True:
    root = '/research/atoms/Session2/'

# root = '/research/atoms/Session1/'
kf_num = 5 # The num when we do K-fold CV
imputed_list = []

for dirpath,dirnames,filenames in os.walk(root):
    for filename in filenames:
        if filename.endswith('_imputed.csv'):
            imputed_list.append(dirpath + '/' + filename)

num_student = len(imputed_list)
print ('The num of students is ', num_student)
count = 0
qualified_list = []
for imputed_file in imputed_list:
    count += 1
    print (count,' ',imputed_file)
    student = pd.read_csv(imputed_file, low_memory=False)
    if len(student['Screen_Number'].unique()) == 1:
        num_student -= 1
        print (imputed_file,'filtering out, screen info missing')
        continue
    if len(student['Response'].unique()) == 1:
        num_student -= 1
        print (imputed_file,'filtering out, response info missing')
        continue
    if len(student['Representation_Number_Within_Screen'].unique()) == 1:
        num_student -= 1
        print (imputed_file,'filtering out, page info missing')
        continue
    qualified_list.append(imputed_file)
    
    if len(qualified % kf_num == 0):
        print('The qualified files can be devided exactly')
    else:
        print('The qualified files CAN NOT be devided exactly, FILTER is needed')
    
kf = KFold(n_splits= kf_num, random_state=None, shuffle=True)
for train_index, test_index in kf.split(qualified_list):
    print ('train_set: ',train_index,'test_set: ', test_index)
print('qualified_list: ',qualified_list)
            
            
            
            
            
            
            
            
            

            
            
            
            
            
            
            
            
            
            
            