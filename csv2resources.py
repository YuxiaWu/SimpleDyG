# 
import pandas as pd
import os
import math
import numpy as np
import random
import sys
import json

# add time token

dataset = sys.argv[1]
timestamp = sys.argv[2]

# save vocab for each timestamp
vocab_path_t = os.path.join('./vocabs/', dataset, str(timestamp))
if not os.path.exists(vocab_path_t):
    os.makedirs(vocab_path_t)

save_path = os.path.join('./resources/', dataset, str(timestamp))

if not os.path.exists(save_path):
    os.makedirs(save_path)

save_file_train = open(os.path.join(save_path,  'train.link_prediction'),'w')
save_file_test = open(os.path.join(save_path,   'test.link_prediction'),'w')
save_file_test_gt = open(os.path.join(save_path,  'test_gt.link_prediction'),'w')
save_file_val = open(os.path.join(save_path, 'val.link_prediction'),'w')
save_file_val_gt = open(os.path.join(save_path,  'val_gt.link_prediction'),'w')

#timestamp = 5
#dataset = 'UCI_13'
data_path = '../all_data/'
# define some parameters
SLICE_LEN = 1024  # the length of each sentence which is feed into the GPT
# define some special tokens
# one sentence be like: <|endoftext|> <|history|> <|shot|>u1 u2 u1 u3 … <|endofshot|> <|shot|>u1 u4 u1 u5<|endofshot|> <|endofhistory|> <|pre|> u1 u2 <endofpre> <|endoftext|> 
BOS = '<|endoftext|>'
EOS = '<|endoftext|>'
HIS = '<|history|>'
EHIS = '<|endofhistory|>'
TIME_ALL = ['<|time'+str(i)+'|>' for i in range(int(timestamp)+1)]
print(TIME_ALL)
PRE = '<|pre|>'
EPRE = '<|endofpre|>'
#shot = '<|shot|>'
#shot_end = '<|endofshot|>'
# read data
if 'UCI' in dataset:
    dataset_read_name = 'uci'
elif 'ML_10M' in dataset:
    dataset_read_name = 'movie'
else:
    dataset_read_name = dataset

data = pd.read_csv(os.path.join(data_path, dataset_read_name, str(timestamp), 'ml_'+ dataset_read_name + '.csv'), index_col=0)

# sort data by ts
data = data.sort_values(by=['ts'])


unique_user_nodes_list = data['u'].unique().tolist()
unique_item_nodes_list = data['i'].unique().tolist()

# train: timestamp
max_timestamp = data['timestamp'].max()
assert max_timestamp == int(timestamp)

if 'dialog' in dataset:
    # for train data, the last timestamp is output data.
    data_train = data[data['timestamp']<=max_timestamp-4] # 0 -->1 

    # val, test data: the output data
    data_val_input = data[data['timestamp']<max_timestamp-2]
    data_val = data[data['timestamp']==max_timestamp-2] # 3,  0 1 2-->3

    data_test_input = data[data['timestamp']<max_timestamp]
    data_test = data[data['timestamp']==max_timestamp] # 5,  0 1 2 3 4 -->5

    # reapeat the data by exchange the "u" and "i
    data_reverse = data_train.copy()

    data_reverse['u'] = data_train['i']
    data_reverse['i'] = data_train['u']

    new_train = pd.concat([data_train, data_reverse], ignore_index=True)

    # sort data by ts
    new_train = new_train.sort_values(by=['ts'])

    val_inputs = pd.concat([data_val_input, data_reverse], ignore_index=True)
    # test_inputs: the history of test
    test_inputs = pd.concat([data_test_input, data_reverse], ignore_index=True)
    full_data = pd.concat([test_inputs, data_test], ignore_index=True)

else:
    data_train = data[data['timestamp']<=max_timestamp-2]
    data_val = data[data['timestamp']==max_timestamp-1]
    data_test = data[data['timestamp']==max_timestamp]

    # reapeat the data by exchange the "u" and "i
    data_reverse = data_train.copy()
    data_reverse['u'] = data_train['i']
    data_reverse['i'] = data_train['u']
    new_train = pd.concat([data_train, data_reverse], ignore_index=True)

    # sort data by ts
    new_train = new_train.sort_values(by=['ts'])
    train_val = pd.concat([new_train, data_val], ignore_index=True)
    full_data = pd.concat([train_val, data_test], ignore_index=True)

    val_inputs = new_train
    test_inputs = train_val



# for training data
train_node_ids = set()
for user_id, user_data in new_train.groupby('u'):
    # how many timestamps user_id has
    num_t = len(user_data['timestamp'].unique())
    min_time_u = int(user_data['timestamp'].min())
    max_time_u = int(user_data['timestamp'].max())

    inputs = BOS + ' ' + HIS + ' ' + str(int(user_id)) +  ' '
    outputs = PRE + ' '
    # if the user only has one timestamp, we only write it into the train file
    if num_t<2:
        item_ids = [int(i) for i in user_data['i'].values.tolist()]
        
        if max_time_u>0:
            TIME_I = TIME_ALL[max_time_u-1]
            inputs+= TIME_I + ' '
        
            TIME_O = TIME_ALL[max_time_u]
            outputs+= TIME_O + ' '

            for item in item_ids[:-1]:
                inputs += str(item) + ' '
            outputs += str(item_ids[-1]) + ' ' + EPRE + ' ' + EOS
            sample = inputs + EHIS + ' ' + outputs
            save_file_train.write(sample +'\n')
    else:
        # the data of max_timestamp is used for output, before max_timestamp is used for input
        inputs_data = user_data[user_data['timestamp']<max_time_u]
        outputs_data = user_data[user_data['timestamp']==max_time_u]
        # get the input data
        # add by time,if there is no data in one time, add the time token
        for i in range(min_time_u, max_time_u):
            inputs += TIME_ALL[i] + ' '
            
            time_data = inputs_data[inputs_data['timestamp']==i]
            if len(time_data)>0:
                for _, row in time_data.iterrows():
                    inputs += str(int(row['i'])) + ' '
        inputs += EHIS + ' '
        # get the output data
        TIME_O = TIME_ALL[max_time_u]
        outputs += TIME_O + ' '
        for _, row in outputs_data.iterrows():
            outputs += str(int(row['i'])) + ' '
        outputs += EPRE + ' ' + EOS
        sample = inputs + outputs
        save_file_train.write(sample +'\n')

# write a function for val and test data
def write_val_test(data, history, save_file, save_file_gt):
    """_summary_
        # for each user_data, the output is the data at predicted timestamp, the input is the data before predicted timestamp
        # for val: the history is the train data. for test, the history is the train_val
    Args:
        data (_type_): _description_
        user_dataindata (_type_): _description_

    """
    for user_id, user_data in data.groupby('u'):
        history_data = history[history['u']==user_id] 
        inputs = BOS + ' ' + HIS + ' ' + str(int(user_id)) +  ' '
        outputs = PRE + ' '
        user_time  = int(user_data['timestamp'].unique().tolist()[0]) # only one timestamp
        # for hepth, the history_data is [] for some users
        if "hepth" in dataset:
            inputs += TIME_ALL[user_time-1] + ' '

        else:
            max_time_u = int(history_data['timestamp'].max())
            min_time_u = int(history_data['timestamp'].min())
        
            # add by time,if there is no data in one time, add the time token
            for i in range(min_time_u, max_time_u+1):
                inputs += TIME_ALL[i] + ' '
                
                time_data = history_data[history_data['timestamp']==i]
                if len(time_data)>0:
                    for _, row in time_data.iterrows():
                        inputs += str(int(row['i'])) + ' ' 

        # get the output data
        TIME_O = TIME_ALL[user_time]
        outputs += TIME_O + ' '

        for _, row in user_data.iterrows():
            outputs += str(int(row['i'])) + ' '
        outputs += EPRE + ' ' + EOS
        save_file.write(inputs + EHIS +'\n')
        save_file_gt.write(outputs+'\n')

write_val_test(data_val, val_inputs, save_file_val, save_file_val_gt)
write_val_test(data_test, test_inputs, save_file_test, save_file_test_gt)

user_item_ids = set(list(full_data['u'])+list(full_data['i']))

'''
assert len(user_item_ids)==len(train_node_ids)
assert len(user_item_ids-set(train_node_ids))==0
assert len(set(train_node_ids)-user_item_ids)==0
'''
# sort 
user_item_ids = list(user_item_ids)
user_item_ids.sort()
# to dict id:id
user_item_ids = {str(i):ind for ind, i in enumerate(user_item_ids)}

# construct new vocab from train_node_ids
#vocab = {str(node_id):ind for ind, node_id in enumerate(train_node_ids)}
# save vocab
#with open(os.path.join(vocab_path_t, 'vocab.json'), 'w') as f:
#    json.dump(vocab, f, indent=4)

with open(os.path.join(vocab_path_t, 'vocab.json'), 'w') as f:
    json.dump(user_item_ids, f, indent=4)

'''
node_features_all = np.load(os.path.join(data_path, dataset_read_name, str(timestamp), 'ml_'+ dataset_read_name + '_node.npy'))
node_features = node_features_all[:len(user_item_ids)]
np.save(os.path.join(save_path,'node_features.npy'), node_features)
'''