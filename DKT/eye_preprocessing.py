# coding: utf-8
import csv
import numpy as np
import pandas as pd
import pdb
import os
import pdb
root = '/research/atoms/Session1/'
#writer = csv.writer(open(root + 'DKT_atoms.csv','w',newline=''))
with open(root + 'Session1_DKT_atoms.csv','w',newline='') as file:
    writer = csv.writer(file)

    imputed_list = []
    for dirpath,dirnames,filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('_imputed.csv'):
                imputed_list.append(dirpath + '/' + filename)

    problem_ID = 0 # make each question to be an ID
    problem_set = {} # unique problem type
    answer_set = {'CORRECT':1, 'INCORRECT':0}
    num_student = len(imputed_list)
    print ('The num of students is ', num_student)
    for imputed_file in imputed_list:
        student = pd.read_csv(imputed_file, low_memory=False)
        Response_null = pd.isnull(student["Response"])
        filter_student = False
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

        #student_res = student[['Screen_Number','Representation_Number_Within_Screen','question','Response']][Response_null == False]
        student_res = student[Response_null == False]
        student_res = student_res[student_res['question'] != '_root'][student_res['question'] != 'done'][student_res['question'] != 'nl'][student_res['Response'] != 'HINT']
        problem_IDs = [] # the total problems sequence of each student.
        answers = [] # the total answers sequence of each student
        for ix, row in student_res.iterrows():
            if row['Stimulus'] != "bl i.jpg":
                continue
            SPquestion = str(row['Representation_Number_Overall']) + row['question']
            # print (SPquestion)
            if not SPquestion in problem_set:
                problem_ID += 1
                problem_set.update({SPquestion: problem_ID})
            problem_IDs.append(problem_set[SPquestion])
            answers.append(answer_set[row['Response']])
        num_problem = len(problem_IDs)
        print (num_problem)
        #if filter_student == True: # Screen or page number missing in this student so we filter it out
            #print (imputed_file,'filtering out, page info missing')
            #num_student -= 1
            #continue
        if imputed_file[-19] == '/':
            writer.writerow([imputed_file[-18:]])
        else:
            writer.writerow([imputed_file[-19:]])
        writer.writerow([num_problem])
        writer.writerow(problem_IDs)
        writer.writerow(answers)

        print(imputed_file)
        #print ('num_problem',num_problem,'\n len(problem_IDs) ',len(problem_IDs),'\n len(answers)', len(answers))
    print ('num_students',num_student)


