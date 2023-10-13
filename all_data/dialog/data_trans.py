
import pandas as pd
import numpy as np
import os

dataset_name = 'dialog'
data_name_ori = './dialog.csv'


data = pd.read_csv(data_name_ori)
print(len(set(list(data['user_id']))))
print(len(set(list(data['item_id']))))

print(data['user_id'].values.max())
print(data['item_id'].values.max())

num_timestamp = max(data['timestamp'])

# let's transform the data into the format of ml_xxx.csv

data['label'] = 0
set1 = set(list(data['user_id'])+list(data['item_id']))


train_node_set = set(list(data['user_id'])+list(data['item_id']))
print(len(train_node_set))
print(len(set(list(data['user_id']))))
print(len(set(list(data['item_id']))))
print(data['user_id'].values.max())
print(data['item_id'].values.max())
set2 = set(list(data['user_id'])+list(data['item_id']))


# each ori_time minus the min one
data['ts'] = data['ori_time'] - min(data['ori_time'])
# sort the data by the ts
data = data.sort_values(by='ts')

data = data[['user_id', 'item_id','ts', 'label',  'timestamp']]
# rename the columns
data.columns = ['u','i','ts','label', 'timestamp']

print('all data')
print('max u id: ', data['u'].values.max()) # 5219
print('max i id: ', data['i'].values.max()) # 7510


for timestamp in range(5, num_timestamp):
    
    cur_data = data[data['timestamp']<=timestamp] #12
    train_data = cur_data[cur_data['timestamp']<timestamp-2] # 0 1 ... 10
    val_test_data = cur_data[cur_data['timestamp']>=timestamp-2] #11 12
    val_data = val_test_data[val_test_data['timestamp']==timestamp-2] #11
    test_data = val_test_data[val_test_data['timestamp']==timestamp] #11
    

    train_node_set = set(list(train_data['u'])+list(train_data['i']))
    print('------------')
    print('timestamp', timestamp)
    # max node id
    print(train_data['u'].values.max())
    print(train_data['i'].values.max())
    
    # for the cur_data, delete the data [u,i] where u or i is not in the train_node_set
    # for hepth, don't delete any data
    new_cur_data = cur_data[cur_data['u'].isin(train_node_set) & cur_data['i'].isin(train_node_set)]
    print('len(new_cur_data)', len(new_cur_data))
    print('len_train_data', len(train_data))
    print('len_val_test_data', len(val_test_data))
    print('len_test_data', len(test_data))
    print('len_val_data', len(val_data))
    new_cur_data['idx'] = range(1,len(new_cur_data)+1)
    new_cur_data.index = range(len(new_cur_data))
    
    save_path = os.path.join('./',str(timestamp))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    new_cur_data.to_csv(os.path.join(save_path, 'ml_'+dataset_name+'.csv'), index=True)

    # get the node features, the node features is np.zeros
    # the nodes are all the  u and i
    nodes = list(set(list(new_cur_data['u'])+list(new_cur_data['i'])))
    feat_dim = 172

    node_features = np.zeros((len(nodes),feat_dim))
    np.save(os.path.join(save_path,'ml_' + dataset_name + '_node.npy'), node_features)

    # get the edge features, the edge features is np.zeros
    edge_features = np.zeros((len(new_cur_data),feat_dim))
    np.save(os.path.join(save_path,'ml_' + dataset_name + '.npy'), edge_features)







