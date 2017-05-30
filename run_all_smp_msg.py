import os

do_not_consider = [10, 14, 18, 23, 26, 28, 33, 38, 49, 4, 9]

for i in range(100):
    if i not in do_not_consider:
        file_name = "atoms" + str(i)
        try:
            os.system("python cleaning_data_msg.py " + file_name)
            os.system("python cleaning_data_smp.py " + file_name)
            print(i)
        except FileNotFoundError:
            continue