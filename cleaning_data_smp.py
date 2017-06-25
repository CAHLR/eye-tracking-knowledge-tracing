import pandas as pd
import numpy as np
import os
import sys

file_name = str(sys.argv[1])

df = pd.read_csv('/research/atoms/good_data/2014 F Session 2/' + file_name + '-eye_data Samples.txt', sep = '\t', error_bad_lines=False, skiprows = 37)
print(df.head())
#aoi_2016 = pd.read_csv('/research/atoms/aoi_2016_v2kc.csv', sep = '|')
identifier = pd.read_csv('/research/atoms/identifier_logMSG_mapping.csv', error_bad_lines=False)

def get_luuid(x):
    return x.split()[1][6:]

identifier['luuid'] = identifier['Event'].apply(get_luuid)
identifier

f = open('/research/atoms/good_data/2014 F Session 2/' + file_name + '-eye_data Samples.txt', 'r')

def get_eye_columns():
    counter = 0
    temp_columns = []
    for line in f.readlines()[37:]:
        if(line[0] == "#"):
            continue
        counter += 1
        if(counter == 1):
            splitted = line.split()
            i = 0
            while i < len(splitted):
                if(splitted[i] == "L" or splitted[i] == "H" or splitted[i] == "R" or splitted[i] == "Pupil"):
                    if(splitted[i] == "R"):
                        print('R reached')
                    #special cases where only one or two letters follow L, H, or R
                    if(splitted[i + 1] == "Validity" or splitted[i + 1] == "Plane" or splitted[i + 1] == "Confidence"):
                        to_append = splitted[i] + "_" + splitted[i + 1]
                        temp_columns.append(to_append)
                        i += 2
                    else:
                        to_append = splitted[i] + "_" + splitted[i + 1] + "_" + splitted[i + 2]
                        temp_columns.append(to_append)
                        if(splitted[i + 1] == "EPOS" or splitted[i + 1] == "GVEC"):
                            i += 3
                        else:
                            #This allows us to skip the [xx] that we might encounter
                            i += 4
                else:
                    temp_columns.append(splitted[i])
                    i += 1
            #print(temp_columns)
        if(counter == 2):
            break
    return temp_columns

eye_columns = get_eye_columns()

pandas_columns = ['username', 'timestamp', 'meta', 'mouseX', 'mouseY', 'Response', 'question', 'stimulus', 'problem',
                  'answer_given',
                  'luuid', 'keypress', 'input_type']

pandas_columns = pandas_columns + eye_columns
output_df = pd.DataFrame(columns = pandas_columns)

smp = df[df['Type'] == "SMP"]
smp.columns = eye_columns
print("SMP Shape: ", smp.shape)
'''for index, row in smp.iterrows():
    #toAdd = row
    toAdd = np.empty((1, 52), dtype = "<U100")[0]
    toAdd[12: 12 + len(eye_columns)] = row
    output_df.loc[index] = toAdd'''


def getRows(row):
    return row
output_df[output_df.columns[13:13+len(smp.columns)]] = smp.apply(getRows, axis = 1)

print("SMP done")
output_df.to_csv(file_name + '_smp.csv')
