import pandas as pd
import os

#file_name must be a string
def generate_csv(file_name):
    atoms10_msg = pd.read_csv(file_name + '_msg.csv')
    atoms10_smp = pd.read_csv(file_name + '_smp.csv')
    atoms10_data = pd.concat([atoms10_msg, atoms10_smp])
    mappings = pd.read_csv('/research/atoms/identifier_logMSG_mapping.csv')

    #columns_for_drop = ['Session', 'condition', 'problem_type', 'representation', 'Stimulus', 'AOI_withPT', 'Step Name']
    def get_luuid(x):
        return x.split()[1][6:]
    #do all the relevant merges using the lookup tables and identifiers
    mappings['luuid'] = mappings['Event'].apply(get_luuid)
    aoi_2k16 = pd.read_csv('/research/atoms/aoi_2016_v2kc.csv', sep = '|')
    aoi_relevant_columns = aoi_2k16[['problem', 'identifier', 'Session', 'condition', 'problem_type', 'representation', 'Stimulus', 'AOI_withPT', 'Step Name',
                                     'KC (ReprAll_fineButAtoms)']]
    merged = atoms10_data.merge(mappings, on="luuid", how = "outer", copy = False)
    final_merge = merged.merge(aoi_relevant_columns, on = 'identifier', how = 'outer', copy = False)
    final_merge = final_merge[pd.isnull(final_merge['Time']) == False]

    final_merge = final_merge.sort_values(by = 'Time')
    final_merge = final_merge.drop('timestamp', axis = 1)
    final_merge = final_merge.drop('problem_x', axis = 1)
    final_merge = final_merge.rename({'problem_y':'problem'})
    #set all rows (both SMP and MSG events) to have the same username. Will make filtering based on student easier
    final_merge['username'] = final_merge['username'].apply(lambda x: file_name)
    final_merge = final_merge.rename(columns={'problem_y': 'problem'})
    final_merge = final_merge.drop(['Event', 'stimulus'], axis=1)
    final_merge.to_csv(file_name + ".csv")
    os.remove(os.getcwd() + '/' + file_name + '_msg.csv')
    os.remove(os.getcwd() + '/' + file_name + '_smp.csv')

    print("Done")


def sort_vals(filename):
    data = pd.read_csv(filename + '.csv')
    data = data.sort_values(by = 'timestamp')
    data.to_csv(filename + '.csv')

files = [4] #, 106, 108, 116, 2, 44, 47, 48, 51, 59, 61]

for i in files:
    #if i not in do_not_consider:
    print("working")
    file_name = "atoms" + str(i)
    try:
        generate_csv(file_name)
        print(i)
    except FileNotFoundError:
        continue