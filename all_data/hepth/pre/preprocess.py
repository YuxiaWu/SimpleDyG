import dill
from collections import defaultdict
from datetime import datetime, timedelta

from scipy.sparse import csr_matrix

import networkx as nx
import numpy as np
import pandas as pd
import json
import time

# paper dates
paper_dates = {}
with open('./hep-th-slacdates') as f:
    lines = f.read().splitlines()
    for l in lines:
        x, y = l.split()
        if x not in paper_dates:
            # y is '1992-12-31'
            # transfer y to timestamp
            timeArry = time.strptime(y, "%Y-%m-%d")

            timestamp = time.mktime(timeArry)
            paper_dates[x] = timestamp      



links = []
ts = []
ctr = 0
node_cnt = 0
node_idx_ori = {}
idx_node = []

links_ori = []

with open('./hep-th-citations') as f:
    lines = f.read().splitlines()
    for l in lines:
        
            
        x, y = l.split()
        # check the time
        x_time = paper_dates[x]
        y_time = paper_dates[y]
        if x_time < y_time:
            continue
        links_ori.append((x,y))

        # x: from paper id  y: to paper id
        t = paper_dates[x] # 946915200.0 
        #756576000
        #datetime.fromtimestamp(777225600)
        timestamp = datetime.fromtimestamp(t) # datetime.datetime(2000, 1, 4, 0, 0)
        year_month = timestamp.strftime("%Y-%m")
        ts.append(timestamp)
        
        ctr += 1
        if ctr % 100000 == 0:
            print (ctr)
            
        if x not in node_idx_ori:
            node_idx_ori[x] = node_cnt 
            node_cnt += 1
            
        if y not in node_idx_ori:
            node_idx_ori[y] = node_cnt 
            node_cnt += 1
    
        links.append((node_idx_ori[x],node_idx_ori[y], timestamp, t, year_month))

print ("Min ts", min(ts), "max ts", max(ts))    
print ("Total time span: {} days".format((max(ts) - min(ts)).days))
links.sort(key =lambda x: x[3])

print('node_cnt: ', node_cnt)
print('edge cnt: ',ctr)
'''
#Min ts 1993-10-01 07:00:00 max ts 2002-03-12 07:00:01
#Total time span: 3084 days
'''

links_df = pd.DataFrame(links)
links_df.columns = ['user_id', 'item_id','timestamp','ori_time', 'year_month']
# let's check about the data size within each year-month
links_df.to_csv('links_df.csv', index=False)



links_df = links_df[links_df['timestamp']>datetime(1993, 5, 1, 0, 0, 0)]
links_df = links_df[links_df['timestamp']<datetime(1995, 5, 1, 0, 0, 0)]
links_df.to_csv('links_df_cut.csv', index=False)


counts = links_df['year_month'].value_counts()
print('counts: ', counts)
mints = links_df['timestamp'].min()
maxts = links_df['timestamp'].max()
print('mints: ', mints)
print('maxts: ', maxts)
print ("Total time span: {} days".format((maxts - mints).days))
print('links_df.shape: ', links_df.shape)


SLICE_DAYS = 30*2
START_DATE = links_df['timestamp'].min() #+ timedelta(240) # datetime.datetime(1993, 11, 30, 7, 0)
END_DATE = links_df['timestamp'].max() #- timedelta(12) # datetime.datetime(2002, 2, 28, 7, 0, 1)

slices_links = defaultdict(lambda : nx.MultiGraph())
slices_features = defaultdict(lambda : {})

print ("Start date", START_DATE)
print ("End date", END_DATE)

print('(END_DATE - START_DATE).days: ', (END_DATE - START_DATE).days) # 725 days

slice_id = 0
# Split the set of links in order by slices to create the graphs. 

#loop over links_df
for index, row in links_df.iterrows():
    a = row['user_id']
    b = row['item_id']
    time = row['timestamp']
    ori_time = row['ori_time']

    prev_slice_id = slice_id
    datetime_object = time

    if datetime_object < START_DATE:
        continue
    if datetime_object > END_DATE:
        break
        days_diff = (END_DATE - START_DATE).days
    else:
        days_diff = (datetime_object - START_DATE).days
        
    #days_diff = (datetime_object - START_DATE).days
        
    slice_id = days_diff // SLICE_DAYS
    
    if slice_id == 1+prev_slice_id and slice_id > 0:
        slices_links[slice_id] = nx.MultiGraph()
        slices_links[slice_id].add_nodes_from(slices_links[slice_id-1].nodes(data=True))
        assert (len(slices_links[slice_id].edges()) ==0)
        #assert len(slices_links[slice_id].nodes()) >0

    if slice_id == 1+prev_slice_id and slice_id ==0:
        slices_links[slice_id] = nx.MultiGraph()

    if a not in slices_links[slice_id]:
        slices_links[slice_id].add_node(a)
    if b not in slices_links[slice_id]:
        slices_links[slice_id].add_node(b)    
    #slices_links[slice_id].add_edge(a,b, weight= e, date=datetime_object)
    slices_links[slice_id].add_edge(a,b, weight= 1, date=ori_time)



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
    print ("Total # nodes", len(node_idx), "max idx", max(node_idx.values()))

    slices_graph_remap = []
    slices_features_remap = []
    for slice_id in slices_graph:
        G = nx.MultiGraph()
        for x in slices_graph[slice_id].nodes():
            G.add_node(node_idx[x])
        for x in slices_graph[slice_id].edges(data=True):
            # x: (7262, 6009, {'weight': 1, 'date': 736358400.0})
            G.add_edge(node_idx[x[0]], node_idx[x[1]], date=x[2]['date'])
        assert (len(G.nodes()) == len(slices_graph[slice_id].nodes()))
        assert (len(G.edges()) == len(slices_graph[slice_id].edges()))
        slices_graph_remap.append(G)
    
    for slice_id in slices_features:
        features_remap = []
        for x in slices_graph_remap[slice_id].nodes():
            features_remap.append(slices_features[slice_id][idx_node[x]])
            #features_remap.append(np.array(slices_features[slice_id][idx_node[x]]).flatten())
        features_remap = csr_matrix(np.squeeze(np.array(features_remap)))
        slices_features_remap.append(features_remap)
    return (slices_graph_remap, slices_features_remap, node_idx)

slices_links_remap, slices_features_remap, node_idx_new = remap(slices_links, slices_features)

# get the node features
# load document_features: dict of {node_id: feature}
import pickle
document_features = pickle.load(open("document_features.pkl", "rb"))
final_remap = {v: k for k, v in node_idx_new.items()}
remap_ori = {v: k for k, v in node_idx_ori.items()}
node_features = []
for idx in range(len(node_idx_new)):
    ori_idx = remap_ori[final_remap[idx]]
    node_features.append(document_features[ori_idx])
node_features = np.array(node_features)
print ("Node features shape", node_features.shape)
# save node_features
np.save("node_features.npy", node_features)


np.savez('graphs.npz', graph=slices_links_remap)
np.savez('features.npz', feats=slices_features_remap)


graphs = np.load("graphs.npz", allow_pickle=True)['graph']
#graphs = slices_links_remap
with open('hepth_ori.csv', 'w') as f:
    
    f.write('user_id,item_id,timestamp,ori_time, state_label,comma_separated_list_of_features\n')
    
    num_time = len(graphs)    
    for timestamp in range(num_time):
        print(timestamp)
        print('edges num: ',len(graphs[timestamp].edges()))

        for (user, item) in nx.Graph(graphs[timestamp]).edges:
            user_time = paper_dates[remap_ori[final_remap[user]]] 
            item_time = paper_dates[remap_ori[final_remap[item]]] 

            if user_time>item_time:
                ori_time = user_time
                user_i = user
                item_i = item
            else:
                ori_time = item_time
                user_i = item
                item_i = user

            
            timestamp = int(timestamp)
            f.write('%d,%d,%d,%d,0,0\n'%(user_i, item_i, timestamp, ori_time))
            #f.write('%d,%d,%d,%d, 0,0\n'%(item, user, timestamp, ori_time))
print('done')



dataset = 'hepth'
data = pd.read_csv(dataset+'_ori.csv')
data2 = data.copy()
data2['time_diff'] = (data['ori_time'] - data['ori_time'].min()) / (data['ori_time'].max() - data['ori_time'].min())
data2 = data2.sort_values(by=['time_diff'])
data2.to_csv(dataset+'.csv', index=False)
# print the max time for each timestamp
for i, group in data2.groupby('timestamp'):
    print(i, group['time_diff'].max())

print(group['time_diff'])