import os

files = [4]#, 106, 108, 116, 2, 44, 47, 48, 51, 59, 61]

for i in files:
    #if i not in do_not_consider:
    file_name = "atoms" + str(i)
    try:
        os.system("python cleaning_data_msg.py " + file_name)
        os.system("python cleaning_data_smp.py " + file_name)
        print(i)
    except FileNotFoundError:
        continue