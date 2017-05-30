import pandas as pd
import numpy as np
import os

import sys

file_name = str(sys.argv[1])

df = pd.read_csv('/research/atoms/raw_data/' + file_name + '-eye_data Samples.txt', sep = '\t', error_bad_lines=False, skiprows = 37)
#print(df.head())
aoi_2016 = pd.read_csv('/research/atoms/aoi_2016_v2kc.csv', sep = '|')
identifier = pd.read_csv('/research/atoms/identifier_logMSG_mapping.csv', error_bad_lines=False)

def get_luuid(x):
    return x.split()[1][6:]

identifier['luuid'] = identifier['Event'].apply(get_luuid)
identifier

f = open('/research/atoms/raw_data/'+ file_name + '-eye_data Samples.txt', 'r')

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
#print("eye_columns", eye_columns)

pandas_columns = ['username', 'timestamp', 'meta', 'mouseX', 'mouseY', 'Response', 'question', 'stimulus', 'problem',
                  'I',
                  'luuid', 'keypress']
columns_for_join = ['Session', 'condition', 'problem_type', 'representation', 'Stimulus', 'AOI_withPT', 'Step Name',
                    'KC (ReprAll_fineButAtoms)']
pandas_columns = pandas_columns + eye_columns + columns_for_join

output_df = pd.DataFrame(columns = pandas_columns)


messages = df[df['Type'] == "MSG"]
messages['L Raw X [px]'] = messages['L Raw X [px]'].str[10:]

def addIdentifier(x):
    splitted = x.split()
    luuid = [i for i in splitted if i[:5] == 'luuid']
    if luuid != []:
        #print(luuid)
        desired_luuid = luuid[0][6:]
        #print(desired_luuid)
        #print(splitted)
        relevant_identifier = identifier[identifier['luuid'] == desired_luuid]
        #get the identifier
        if relevant_identifier.empty == False:
            identifier_wanted = relevant_identifier.reset_index()['identifier'][0]
            return identifier_wanted
        else:
            #print('No match')
            return

messages['identifier'] = messages['L Raw X [px]'].apply(addIdentifier)

def check_luuid(row, output, output_index):
    splitted = row['L Raw X [px]'].split()
    #create a list containing luuid term if luuid is present
    luuid = [i for i in splitted if i[:5] == 'luuid']
    if luuid != []: #True if luuid is present, false otherwise
        output[output_index] = luuid[0][6:]
        #print(len(output[output_index]))
    return output



def join_aoi_output(output, identifier):
    investigating_index = 0
    for index, row in output.iterrows():

        luuid = row['luuid']
        toAdd = np.empty((1, 46), dtype="<U100")[0]
        if (luuid != ""):
            relevant_identifier = identifier[identifier['luuid'] == luuid]
            # get the identifier
            if relevant_identifier.empty == False:
                # print("Inside")
                identifier_found = relevant_identifier.reset_index()['identifier'][0]
                # find row where the corresponding identifier is present
                relevant_aoi = aoi_2016[aoi_2016['identifier'] == identifier_found]
                relevant_aoi = relevant_aoi.reset_index()
                # now do join?? i need to add identifier columns to that
                if relevant_aoi.empty == False:
                    # print("Inside 2")
                    investigating_index = index

                    toAdd = np.array(output.loc[index])
                    output['Session'][index] = relevant_aoi['Session'][0]
                    output['problem'][index] = relevant_aoi['problem'][0]
                    output['condition'][index] = relevant_aoi['condition'][0]
                    output['problem_type'][index] = relevant_aoi['problem_type'][0]
                    output['representation'][index] = relevant_aoi['representation'][0]
                    output['Stimulus'][index] = relevant_aoi['Stimulus'][0]
                    output['AOI_withPT'][index] = relevant_aoi['AOI_withPT'][0]
                    output['Step Name'][index] = relevant_aoi['Step Name'][0]
                    output['KC (ReprAll_fineButAtoms)'][index] = relevant_aoi['KC (ReprAll_fineButAtoms)'][0]
                    # print(output.loc[index]) #= toAdd
    #print(output.loc[investigating_index])
    return output


#code for processing keypress
keypressed = messages[messages['L Raw X [px]'].str[:12] == " UE-keypress"]
counter = 0
for index, row in keypressed.iterrows():
    '''counter += 1
    print(row['Time'])
    if(counter == 2):
        break'''
    toAdd = np.empty((1, len(output_df.columns)), dtype = "<U100")[0]
    #toAdd[:] = np.NAN
    toAdd[0] = 'atoms10'
    toAdd[1] = row['Time']
    #add to keypress column
    toAdd[11] = row['L Raw X [px]'][-1]
    toAdd[12] = row['Time']
    #Event Type
    toAdd[13] = 'UE-keypress'
    #add the luuids if present
    toAdd = check_luuid(row, toAdd, 10)
    #print("Output_df column length", len(output_df.columns))
    #print("Output_df column length", len(toAdd))
    output_df.loc[index] = toAdd

'''answers = messages[messages['L Raw X [px]'].str[:10] == " LOG luuid"]
splitted = answers.loc[1228807]['L Raw X [px]'].split()#[5][5:]
#any("luuid" in s for s in splitted)
splitted#[1][6:]
indices = [i for i in splitted if i[:6] == 'luuid']

answers = messages[messages['L Raw X [px]'].str[:10] == " LOG luuid"]
row = answers.loc[155583]
splitted = row['L Raw X [px]'].split()

row = answers.loc[1211153]
splitted = row['L Raw X [px]'].split()

row = answers.loc[1208616]
splitted = row['L Raw X [px]'].split()
splitted, splitted[-2]'''

#toAdd[10] = messages

#code for processing answers
answers = messages[messages['L Raw X [px]'].str[:10] == " LOG luuid"]
for index, row in answers.iterrows():
    '''counter += 1
    print(row['Time'])
    if(counter == 2):
        break'''
    toAdd = np.empty((1, len(output_df.columns)), dtype = "<U100")[0]
    #toAdd[:] = np.NAN
    toAdd[0] = 'atoms10'
    toAdd[1] = row['Time']
    splitted = row['L Raw X [px]'].split()
    if('Eval' in row['L Raw X [px]']):
        response = splitted[5][5:]
        toAdd[5] = response
        question = splitted[6][2:]
        toAdd[6] = question
        #Event Type
        toAdd[13] = 'ButtonPressed'
    else:
        #print("Splitted", splitted)
        try:
            question = splitted[5][2:]
            toAdd[6] = question
        #Event Type

        except IndexError:
            continue
        toAdd[13] = 'UpdateCheckBox'
    #add the 'I' column
    toAdd[9] = splitted[-2]
    toAdd[12] = row['Time']
    #add the luuids if present
    toAdd = check_luuid(row, toAdd, 10)
    output_df.loc[index] = toAdd


#code for LOG_MSG_INIT
Log_msg = messages[messages['L Raw X [px]'].str[:13] == " LOG_MSG_INIT"]


for index, row in Log_msg.iterrows():
    toAdd = np.empty((1, len(output_df.columns)), dtype = "<U100")[0]
    toAdd[0] = 'atoms10'
    toAdd[1] = row['Time']
    splitted = row['L Raw X [px]'].split()
    '''if('Eval' in row['L Raw X [px]']):
        response = splitted[5][5:]
        toAdd[4] = response
        question = splitted[6][2:]
        toAdd[5] = question
        print(toAdd)
    else:
        question = splitted[5][2:]
        toAdd[5] = question'''
    #toAdd[8] = row['L Raw X [px]'].split()[1][6:]
    toAdd[12] = row['Time']
    #Event Type
    toAdd[13] = "LOG_MSG_INIT"
    #add the luuids if present
    toAdd = check_luuid(row, toAdd, 10)
    output_df.loc[index] = toAdd

#code for mouseclick
mouse_click = messages[messages['L Raw X [px]'].str[:14] == " UE-mouseclick"]
mouse_click_filtered = mouse_click[mouse_click['L Raw X [px]'].str[15:19] == 'left']

for index, row in mouse_click_filtered.iterrows():
    toAdd = np.empty((1, len(output_df.columns)), dtype = "<U100")[0]
    toAdd[0] = 'atoms10'
    toAdd[1] = row['Time']
    splitted = row['L Raw X [px]'].split()
    '''if('Eval' in row['L Raw X [px]']):
        response = splitted[5][5:]
        toAdd[4] = response
        question = splitted[6][2:]
        toAdd[5] = question
        print(toAdd)
    else:
        question = splitted[5][2:]
        toAdd[5] = question'''
    #add mouseX
    toAdd[3] = row['L Raw X [px]'].split()[2][2:6]
    #add mouseY
    toAdd[4] = row['L Raw X [px]'].split()[3][2:6]
    toAdd[12] = row['Time']
    #Event Type
    toAdd[13] = "UE-mouseclick"
    #add the luuids if present
    toAdd = check_luuid(row, toAdd, 10)
    output_df.loc[index] = toAdd

# code for WebsiteMargin
website_margin = messages[messages['L Raw X [px]'].str[:19] == " fullWebsite Margin"]

for index, row in website_margin.iterrows():
    toAdd = np.empty((1, len(output_df.columns)), dtype="<U100")[0]
    toAdd[0] = 'atoms10'
    toAdd[1] = row['Time']
    splitted = row['L Raw X [px]'].split()
    # add websiteMargin to meta column
    toAdd[2] = row['L Raw X [px]'][20:]

    toAdd[12] = row['Time']
    # Event Type
    toAdd[13] = "fullWebsite Margin"
    # add the luuids if present
    toAdd = check_luuid(row, toAdd, 10)
    output_df.loc[index] = toAdd

scroll =messages[messages['L Raw X [px]'].str[:7] == " scroll"]

# code for scroll
scroll = messages[messages['L Raw X [px]'].str[:7] == " scroll"]
for index, row in scroll.iterrows():
    toAdd = np.empty((1, len(output_df.columns)), dtype="<U100")[0]
    toAdd[0] = 'atoms10'
    toAdd[1] = row['Time']
    splitted = row['L Raw X [px]'].split()
    # add data following word 'scroll' to meta column
    toAdd[2] = row['L Raw X [px]'][8:]

    toAdd[12] = row['Time']
    # Event Type
    toAdd[13] = "scroll"
    # add the luuids if present
    toAdd = check_luuid(row, toAdd, 10)
    output_df.loc[index] = toAdd

#code for snapshot
scroll = messages[messages['L Raw X [px]'].str[-3:] == "jpg"]
for index, row in scroll.iterrows():
    toAdd = np.empty((1, len(output_df.columns)), dtype = "<U100")[0]
    toAdd[0] = 'atoms10'
    toAdd[1] = row['Time']
    #splitted = row['L Raw X [px]'].split()
    #add data following word 'scroll' to meta column
    toAdd[2] = str(row['L Raw X [px]'][:]) #+ ".jpg"
    #print(toAdd[1])
    #Event Type
    toAdd[13] = "jpg"
    toAdd[12] = row['Time']
    #add the luuids if present
    toAdd = check_luuid(row, toAdd, 10)
    output_df.loc[index] = toAdd

#code for EndTrial
scroll = messages[messages['L Raw X [px]'].str[:9] == " EndTrial"]
for index, row in scroll.iterrows():
    toAdd = np.empty((1, len(output_df.columns)), dtype = "<U100")[0]
    toAdd[0] = 'atoms10'
    toAdd[1] = row['Time']
    #splitted = row['L Raw X [px]'].split()
    #add info for EndTrial to meta column
    toAdd[2] = str(row['L Raw X [px]'][:]) #+ ".jpg"
    #print(toAdd[1])
    toAdd[12] = row['Time']
    #Event Type
    toAdd[13] = "EndTrial"
    #add the luuids if present
    toAdd = check_luuid(row, toAdd, 10)
    output_df.loc[index] = toAdd



print("MSG Done")



'''for index, row in df.iterrows():

    output_df.loc[index]['luuid'] = row['luuid']'''

'''def getLuuid(row):
    return row['luuid']

output_df['luuid'] = df.apply(getLuuid, axis = 1)'''

output_df.to_csv(file_name + '_msg.csv')
