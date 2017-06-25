import pandas as pd
import numpy as np

def impute_smp(df):
    curr_l_raw_x = np.nan
    curr_l_raw_y = np.nan
    curr_l_dia_x = np.nan
    curr_l_dia_y = np.nan
    curr_l_mapped_diameter = np.nan
    curr_pupil_confidence = np.nan
    l_raw_x_array = np.empty((len(df.index, )))
    l_raw_y_array = np.empty((len(df.index, )))
    l_dia_x_array = np.empty((len(df.index, )))
    l_dia_y_array = np.empty((len(df.index, )))
    #l_mapped_diameter_array = np.empty((len(df.index, )))
    pupil_confidence_array = np.empty((len(df.index, )))
    for index, row in df.iterrows():
        #if one of the eye-tracking events is NaN, all others will be NaNs as well
        if pd.isnull(df.loc[index]['L_Raw_X']):
            if pd.isnull(curr_l_raw_x) == False:
                l_raw_x_array[index] = curr_l_raw_x
                l_raw_y_array[index] = curr_l_raw_y
                l_dia_x_array[index] = curr_l_dia_x
                l_dia_y_array[index] = curr_l_dia_y
                #l_mapped_diameter_array[index] = curr_l_mapped_diameter
                pupil_confidence_array[index] = curr_pupil_confidence
            else:
                continue
        else:
            l_raw_x_array[index] = df.loc[index]['L_Raw_X']
            curr_l_raw_x = df.loc[index]['L_Raw_X']
            l_raw_y_array[index] = df.loc[index]['L_Raw_Y']
            curr_l_raw_y = df.loc[index]['L_Raw_Y']
            l_dia_x_array[index] = df.loc[index]['L_Dia_X']
            curr_l_dia_x = df.loc[index]['L_Dia_X']
            l_dia_y_array[index] = df.loc[index]['L_Dia_Y']
            curr_l_dia_y = df.loc[index]['L_Dia_Y']
            pupil_confidence_array[index] = df.loc[index]['Pupil_Confidence']
            curr_pupil_confidence = df.loc[index]['Pupil_Confidence']
    df['L_Raw_X'] = l_raw_x_array
    df['L_Raw_Y'] = l_raw_y_array
    df['L_Dia_X'] = l_dia_x_array
    df['L_Dia_Y'] = l_dia_y_array
    #df['L_Mapped_Diameter'] = l_mapped_diameter_array
    df['Pupil_Confidence'] = pupil_confidence_array
    return df


def impute_msg(df, column_name):
    times = df[df['Type'] == 'scroll']['Time']
    print(times)
    column_array = np.empty((len(df.index, )), dtype=object)
    for i in range(1, len(np.array(times.index)), 2):
        print(i)
        try:
            curr_df = df[(df['Time'] >= times.iloc[i]) & (df['Time'] < times.iloc[i + 1])]
            print(curr_df[column_name].unique())
            print(curr_df['problem_type'].unique())

            if len(curr_df[column_name].unique()) == 1:
                if np.isnan(curr_df[column_name].unique()[0]):
                    continue
                else:
                    new_column = curr_df[column_name].unique()[0]
                    for index, elem in curr_df.iterrows():
                        column_array[index] = new_column

            elif len(curr_df[column_name].unique()) == 2:
                new_column = curr_df[column_name].unique()[1]
                # print(df[(df['Time'] >= times.iloc[i]) & (df['Time'] < times.iloc[i + 2])])
                for index, elem in curr_df.iterrows():
                    column_array[index] = new_column
                    # print(column_array[index] )
                    # print(df.loc[index][column_name])
            else:
                print("Wrong column passed in")
        except IndexError:
            # print(df[df['Time'] >= times.iloc[i]][column_name].unique())
            # print(df[df['Time'] >= times.iloc[i]]['problem_type'].unique())
            curr_df = df[df['Time'] >= times.iloc[i]]
            if len(curr_df[column_name].unique()) == 1:
                continue
            new_column = curr_df[column_name].unique()[1]
            for index, elem in curr_df.iterrows():
                column_array[index] = new_column
    df[column_name] = column_array
    return df


def impute_representation(df, column_name):
    times = df[df['Type'] == 'scroll']['Time']
    print(times)
    column_array = np.empty((len(df.index, )), dtype=object)
    for i in range(1, len(np.array(times.index)), 2):
        try:
            curr_df = df[(df['Time'] >= times.iloc[i]) & (df['Time'] < times.iloc[i + 1])]
        except IndexError:
            curr_df = df[df['Time'] >= times.iloc[i]]

        unique_vals = curr_df[column_name].unique()

        start_time = times.iloc[i]
        try:
            most_recent = unique_vals[1]
        except IndexError:
            continue
        first_rep_occurence = curr_df[curr_df[column_name] == unique_vals[1]]['Time'].iloc[0]
        print("first_rep_occurence", first_rep_occurence)
        for index, _ in df[(df['Time'] >= start_time) & (df['Time'] <= first_rep_occurence)].iterrows():
            column_array[index] = most_recent

        for index, elem in df[(df['Time'] >= first_rep_occurence) & (df['Time'] < times.iloc[i + 2])].iterrows():
            # print(elem[column_name], pd.isnull(elem[column_name]))
            if pd.isnull(elem[column_name]) == False and elem[column_name] != most_recent:
                print(elem[column_name])
                most_recent = elem[column_name]
            column_array[index] = most_recent

            # print(column_array[index])
    print(column_array)
    df[column_name] = column_array
    return df

atoms4 = pd.read_csv('atoms4.csv')
atoms4 = impute_msg(atoms4, 'Stimulus')
atoms4 = impute_msg(atoms4, 'problem_type')
atoms4 = impute_representation(atoms4, 'representation')
atoms4 = impute_smp(atoms4)
##remove these two
atoms4 = atoms4.rename(columns = {'problem_y':'problem'})
atoms4 = atoms4.drop(['Event', 'stimulus'], axis = 1)
atoms4.to_csv('atoms4_imputed.csv')
