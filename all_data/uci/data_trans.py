
import pandas as pd
import numpy as np
import os

dataset_name = 'uci'
data_name_ori = 'UCI_13_ori.csv'

data = pd.read_csv(data_name_ori)
print(len(set(list(data['user_id']))))
print(len(set(list(data['item_id']))))

print(data['user_id'].values.max())
print(data['item_id'].values.max())
data['label'] = 0

set1 = set(list(data['user_id'])+list(data['item_id']))

data = data.iloc[::2,:]

train_node_set = set(list(data['user_id'])+list(data['item_id']))
print(len(train_node_set))
print(len(set(list(data['user_id']))))
print(len(set(list(data['item_id']))))
print(data['user_id'].values.max())
print(data['item_id'].values.max())
set2 = set(list(data['user_id'])+list(data['item_id']))


data['ts'] = data['ori_time'] - min(data['ori_time'])
data = data.sort_values(by='ts')

data = data[['user_id', 'item_id','ts', 'label',  'timestamp']]
data.columns = ['u','i','ts','label', 'timestamp']


for timestamp in range(3,13):
    
    cur_data = data[data['timestamp']<=timestamp] #12
    train_data = cur_data[cur_data['timestamp']<timestamp-1] # 0 1 ... 10
    val_test_data = cur_data[cur_data['timestamp']>=timestamp-1] #11 12
    

    train_node_set = set(list(train_data['u'])+list(train_data['i']))

    new_cur_data = cur_data[cur_data['u'].isin(train_node_set) & cur_data['i'].isin(train_node_set)]

    new_cur_data['idx'] = range(1,len(new_cur_data)+1)
    new_cur_data.index = range(len(new_cur_data))

    save_path = os.path.join('./',str(timestamp))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    new_cur_data.to_csv(os.path.join(save_path, 'ml_'+dataset_name+'.csv'), index=True)

    # get the node features, the node features is np.zeros
    # the nodes are all the  u and i
    nodes = list(set(list(new_cur_data['u'])+list(new_cur_data['i'])))
    # np.zeros, save the node features to ml_uci_13_node.npy
    feat_dim = 172
    node_features = np.zeros((len(nodes),feat_dim))
    np.save(os.path.join(save_path,'ml_' + dataset_name + '_node.npy'), node_features)

    # get the edge features, the edge features is np.zeros
    # np.zeros, save the edge features to ml_uci_13.npy
    edge_features = np.zeros((len(new_cur_data),feat_dim))
    np.save(os.path.join(save_path,'ml_' + dataset_name + '.npy'), edge_features)







