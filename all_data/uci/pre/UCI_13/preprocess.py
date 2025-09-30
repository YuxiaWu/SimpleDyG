from collections import defaultdict
from datetime import datetime, timedelta

from scipy.sparse import csr_matrix

import networkx as nx
import numpy as np
import pandas as pd
import json

links = []
ts = []
ctr = 0
node_cnt = 0
node_idx = {}
idx_node = []


with open('./out.opsahl-ucsocial') as f:
    lines = f.read().splitlines()
    for l in lines:
        if l[0] == '%':
            continue
            
        x, y, e, t = map(int, l.split())
        # print (x,y,e,t)
        timestamp = datetime.fromtimestamp(t)
        ts.append(timestamp)
        
        ctr += 1
        if ctr % 100000 == 0:
            print (ctr)
            
        if x not in node_idx:
            node_idx[x] = node_cnt 
            node_cnt += 1
            
        if y not in node_idx:
            node_idx[y] = node_cnt 
            node_cnt += 1
    
        links.append((node_idx[x],node_idx[y], timestamp,t))

print ("Min ts", min(ts), "max ts", max(ts))    
print ("Total time span: {} days".format((max(ts) - min(ts)).days))
links.sort(key =lambda x: x[2])


links_df = pd.DataFrame(links)
links_df.columns = ['user_id', 'item_id','timestamp','ori_time']

links_df['ori_diff'] = (links_df['ori_time'] - links_df['ori_time'].min()) 

links_df['diff_norm'] = (links_df['ori_time'] - links_df['ori_time'].min()) / (links_df['ori_time'].max() - links_df['ori_time'].min())
links_df.to_csv('links_df.csv', index=False)



SLICE_DAYS = 10
START_DATE = min(ts) + timedelta(5)
END_DATE = max(ts) - timedelta(60)

slices_links = defaultdict(lambda : nx.MultiGraph())
slices_features = defaultdict(lambda : {})

print ("Start date", START_DATE)
print ("End date", END_DATE)

slice_id = 0
for (a, b, time, ori_time) in links:
    prev_slice_id = slice_id
    datetime_object = time
    if datetime_object < START_DATE:
        continue
    if datetime_object > END_DATE:
        break
        days_diff = (END_DATE - START_DATE).days
    else:
        days_diff = (datetime_object - START_DATE).days
        
    
    slice_id = days_diff // SLICE_DAYS
    
    if slice_id == 1+prev_slice_id and slice_id > 0:
        slices_links[slice_id] = nx.MultiGraph()
        slices_links[slice_id].add_nodes_from(slices_links[slice_id-1].nodes(data=True))
        assert (len(slices_links[slice_id].edges()) ==0)

    if slice_id == 1+prev_slice_id and slice_id ==0:
        slices_links[slice_id] = nx.MultiGraph()

    if a not in slices_links[slice_id]:
        slices_links[slice_id].add_node(a)
    if b not in slices_links[slice_id]:
        slices_links[slice_id].add_node(b)    
    slices_links[slice_id].add_edge(a,b, date=ori_time)
    
for slice_id in slices_links:
    print ("# nodes in slice", slice_id, len(slices_links[slice_id].nodes()))
    print ("# edges in slice", slice_id, len(slices_links[slice_id].edges()))
    
    temp = np.identity(len(slices_links[max(slices_links.keys())].nodes()))
    print ("Shape of temp matrix", temp.shape)
    slices_features[slice_id] = {}
    for idx, node in enumerate(slices_links[slice_id].nodes()):
        slices_features[slice_id][node] = temp[idx]


def remap(slices_graph, slices_features):
    all_nodes = []
    for slice_id in slices_graph:
        assert len(slices_graph[slice_id].nodes()) == len(slices_features[slice_id])
        all_nodes.extend(slices_graph[slice_id].nodes())
    # if 3 in all_nodes:
    #     print('Yes')
    all_nodes = list(set(all_nodes))
    print ("Total # nodes", len(all_nodes), "max idx", max(all_nodes))
    ctr = 0
    node_idx = {}
    idx_node = []
    for slice_id in slices_graph:
        for node in slices_graph[slice_id].nodes():
            if node not in node_idx:
                node_idx[node] = ctr
                idx_node.append(node)
                ctr += 1
    slices_graph_remap = []
    slices_features_remap = []
    for slice_id in slices_graph:
        G = nx.MultiGraph()
        for x in slices_graph[slice_id].nodes():
            G.add_node(node_idx[x])
        for x in slices_graph[slice_id].edges(data=True):
            G.add_edge(node_idx[x[0]], node_idx[x[1]], date=x[2]['date'])
        assert (len(G.nodes()) == len(slices_graph[slice_id].nodes()))
        assert (len(G.edges()) == len(slices_graph[slice_id].edges()))
        slices_graph_remap.append(G)
    
    import time
    start_time = time.time()
    for slice_id in slices_features:
        features_remap = []
        row_idx_lst = []
        col_idx_lst = []
        value_lst = []

        for x in slices_graph_remap[slice_id].nodes():
            # 找出非零元素并储存index，用(x, index)定位元素位置，元素值也一并存储，不存储稠密矩阵
            row_idx = np.nonzero(slices_features[slice_id][idx_node[x]])[0]
            row_idx_lst = np.concatenate((row_idx_lst, row_idx))
            value = slices_features[slice_id][idx_node[x]][row_idx]
            value_lst = np.concatenate((value_lst, value))
            col_idx = np.repeat(x, len(row_idx))
            col_idx_lst = np.concatenate((col_idx_lst, col_idx))
        
        len_feature = len(slices_features[slice_id][idx_node[x]])
        features_remap = csr_matrix((value_lst, (row_idx_lst, col_idx_lst)), shape=(len(row_idx_lst), len_feature))
        slices_features_remap.append(features_remap)
    end_time = time.time()
    run_time = end_time - start_time
    print("run_time:", run_time)

    return (slices_graph_remap, slices_features_remap, node_idx, idx_node)
print('Remapping... ')
slices_links_remap, slices_features_remap, node_idx, idx_node = remap(slices_links, slices_features)
print('Remap done!')
graphs = slices_links_remap
print('Writing the ori csv... ')
with open('UCI_13_ori.csv', 'w') as f:
    f.write('user_id,item_id,timestamp,ori_time, state_label,comma_separated_list_of_features\n')
    
    num_time = len(graphs)    
    for timestamp in range(num_time):
        for (user, item) in nx.Graph(graphs[timestamp]).edges:
            ori_time = nx.Graph(graphs[timestamp]).edges[user, item]['date']
            ori_time = int(ori_time)
            user = int(user)
            item = int(item)
            timestamp = int(timestamp)
            f.write('%d,%d,%d,%d,0,0\n'%(user, item, timestamp, ori_time))
            f.write('%d,%d,%d,%d, 0,0\n'%(item, user, timestamp, ori_time))
print('Writing done')

data = pd.read_csv('UCI_13_ori.csv')
data['ori_time'] = (data['ori_time'] - data['ori_time'].min()) / (data['ori_time'].max() - data['ori_time'].min())
print('Sorting ... ')
data = data.sort_values(by=['ori_time'])
data.to_csv('UCI_13.csv', index=False)
for i, timestamp in data.groupby('timestamp'):
    print(i, timestamp['ori_time'].max())