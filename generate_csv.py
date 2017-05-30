import pandas as pd
import os

#file_name must be a string
def generate_csv(file_name):
    atoms10_msg = pd.read_csv(file_name + '_msg.csv')
    atoms10_smp = pd.read_csv(file_name + '_smp.csv')
    atoms10_data = pd.concat([atoms10_msg, atoms10_smp])
    mappings = pd.read_csv('/research/atoms/identifier_logMSG_mapping.csv')
    def get_luuid(x):
        return x.split()[1][6:]
    #do all the relevant merges using the lookup tables and identifiers
    mappings['luuid'] = mappings['Event'].apply(get_luuid)
    aoi_2k16 = pd.read_csv('/research/atoms/aoi_2016_v2kc.csv', sep = '|')
    aoi_relevant_columns = aoi_2k16[['identifier', 'Session', 'condition', 'problem_type', 'representation', 'Stimulus', 'AOI_withPT', 'Step Name']]
    merged = atoms10_data.merge(mappings, on="luuid", how = "outer")
    final_merge = merged.merge(aoi_relevant_columns, on = 'identifier', how = 'outer')
    final_merge.to_csv(file_name + ".csv")
    os.remove(os.getcwd() + '/' + file_name + '_msg.csv')
    os.remove(os.getcwd() + '/' + file_name + '_smp.csv')

    print("Done")


do_not_consider = [10, 14, 18, 23, 26, 28, 33, 38, 49, 4, 9]

for i in range(100):
    if i not in do_not_consider:
        file_name = "atoms" + str(i)
        try:
            generate_csv(file_name)
            print(i)
        except FileNotFoundError:
            continue