avg_acc = []
cv_count = 0
for indexes in cross_val_list:
    cv_count += 1
    print ('cv_count = ',cv_count)
    train_indexes, val_indexes = indexes
    train_input = []
    train_truth = []
    val_input = []
    val_truth = []


    '''Creating y_order matrix for all modes'''
    train_order = []
    val_order = []
    acc = []
    for val_index in val_indexes:
        time_seq_answer = answer_dict[train_val_data[val_index]]
        time_seq_order = order_dict[train_val_data[val_index]]
        seq_answer = time_seq_answer[time_seq_answer!=-1]
        seq_order = time_seq_order[time_seq_answer!=-1]
        seq_table = pd.DataFrame([seq_order,seq_answer])
        seq_table = seq_table.transpose()
        
        #y_order_majority 
        grouped = seq_table.groupby(seq_table['SRquestion_ID'])
        for i,content in grouped:
            if (sum(content.Response)/len(content.Response))<0.5:
                seq_table.loc[seq_table.SRquestion_ID==i,'Response'] = 1- \
                seq_table.loc[seq_table.SRquestion_ID==i,'Response']
                #content.Response = 1 - content.Response
        acc.append(sum(seq_table.Response)/len(seq_table.Response))
    acc = sum(acc)/len(acc)
    avg_acc.append(acc)
        

    