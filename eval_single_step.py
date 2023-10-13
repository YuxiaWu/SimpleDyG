
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
#from utils.args_parser import ArgsParser
from data.dataset.multiwoz import MultiWozDataset
#from evaluate_multiwoz import MultiWozDB
from utils.multiwoz import dbPointer
from utils.simpletod import *
from utils.Evaluation import Evaluation

import json
import ipdb
import sys
import os
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
import numpy as np
import pandas as pd

from utils.gpt2_args_parser import ArgsParser
args = ArgsParser().parse()

dataset = args.dataset
timestamp = args.timestamp
checkpoint_path = args.output_dir

args.para_names = ['dataset', 'method', 'time', 'nlayer','nhead','nemb','bz','lr','seed']
args.para_values = [args.dataset, 'SimpleDyG', args.timestamp, args.n_layer, args.n_head, args.n_embed, args.per_gpu_train_batch_size, args.learning_rate, args.seed]
    
spl_tokens = ['<|history|>','<|endofhistory|>','<|pre|>','<|endofpre|>','<|endoftext|>','[PAD]'] + ['<|time'+str(i)+'|>' for i in range(int(args.timestamp)+1)]    
file_path = os.path.join('resources', dataset, timestamp,'test.link_prediction')
file_path_gt = os.path.join('resources', dataset, timestamp, 'test_gt.link_prediction')

with open(file_path, encoding="utf-8") as f:
    data = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

with open(file_path_gt, encoding="utf-8") as f:
    data_gt = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

# make sure the length of data and data_gt are the same
assert len(data) == len(data_gt)

save_score_path = os.path.join('results-jac', dataset, timestamp, 'test_score')
if not os.path.exists(save_score_path):
    os.makedirs(save_score_path)

save_score = open(os.path.join(save_score_path, 'score_all.txt'),'a')

test_user_ids_DySAT = None
# read the vocab of this timestamp, when evluating, omit the ids not in the vocab
vocab_file = os.path.join('./vocabs', dataset, timestamp, 'vocab.json')
vocab = json.load(open(vocab_file, 'r'))
max_score = -1
Eval = Evaluation()
steps = [0] #[steps[-1]]
for ind_step,step in enumerate(steps):
    model_checkpoint = os.path.join(checkpoint_path, 'checkpoint-{}'.format(step))
    print('model_checkpoint: ', model_checkpoint)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_checkpoint,'tokenizer.json'), use_fast=False)
    model = GPT2LMHeadModel.from_pretrained(model_checkpoint)

    model.eval()
    model.to('cuda')

    break_tokens = tokenizer.encode("<|endoftext|>")
    MAX_LEN = model.config.n_ctx

    generated_dict = {}
    num_data = len(data)
    print('num_data: ', num_data)

    topk = [1, 5]#, 10, 20, 50, 100]
    metric_terms = ['NDCG','jaccard']
    top_k_scores = {metric: len(topk)*[0] for metric in metric_terms}

    # loop through data and data_gt
    num_user_test = 0
    for i, (input_text, text_gt) in enumerate(tqdm(zip(data, data_gt))):
        generated_dict[i] = {}
        user_id = input_text.split()[2]
        target_list = text_gt.split()[1:-2]
        # remove all user id in target_list
        target_list = [token for token in target_list if token != user_id]
        # get the target id appaer in the vocab
        target_list = [token for token in target_list if token in vocab]

        if len(target_list) == 0:
            print('text_gt: ', text_gt)
            continue
        indexed_tokens = tokenizer.encode(input_text)
        num_user_test+=1 
        if len(indexed_tokens) > MAX_LEN:
            print('len_input: ', len(indexed_tokens))
            indexed_tokens = indexed_tokens[-1000:]

        tokens_tensor = torch.tensor([indexed_tokens])

        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to('cuda')
        predicted_index = []
        len_input = len(indexed_tokens)
        while predicted_index not in break_tokens:
            outputs = model(tokens_tensor)
            predictions = outputs[0]
            predicted_index = torch.argmax(predictions[0, -1, :]).item() 
            indexed_tokens += [predicted_index] 
            
            predicted_text = tokenizer.decode(indexed_tokens)
            
            tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
            if len(indexed_tokens) >= MAX_LEN-len(spl_tokens):
                break
            if tokenizer.decode(indexed_tokens).endswith('<|endoftext|>'):
                break
            #if tokenizer.decode(indexed_tokens).endswith('[PAD]'):
            #    break 

        predicted_text = tokenizer.decode(indexed_tokens)
        predicted_list = predicted_text.split()
        predicted_list = predicted_list[len_input:]

        # remove all user id in predicted_list,remove the tokens if it belongs to spl_token
        predicted = [token for token in predicted_list if token != user_id]
        predicted = [token for token in predicted if token not in spl_tokens]
        
        # ndcg
        if 'NDCG' in metric_terms:
            for topi, k in enumerate(topk):
                result = Eval.ndcg_k(predicted, target_list, k)
                top_k_scores['NDCG'][topi] += result  
        if 'jaccard' in metric_terms:
            result = Eval.jaccard(predicted, target_list)
            for topi, k in enumerate(topk):
                top_k_scores['jaccard'][topi] += result

        generated_dict[i]['user_id'] = user_id
        generated_dict[i]['input'] = input_text
        generated_dict[i]['target_list'] = target_list
        generated_dict[i]['len input_text'] = len(input_text.split())
        generated_dict[i]['predicted_list_ori'] = predicted_list
        generated_dict[i]['predicted'] = predicted
        generated_dict[i]['NDCG@k'] = [str(k) for k in top_k_scores['NDCG']]
        generated_dict[i]['num_user_test'] = str(num_user_test)


    print('num_user_test: ', num_user_test)
    for metric in metric_terms:
        for topi, k in enumerate(topk):
            # keep 4 digits after decimal point
            top_k_scores[metric][topi] = round(top_k_scores[metric][topi] / num_user_test, 4)

    ndcg1 = top_k_scores['NDCG'][0]
    if ndcg1 > max_score:
        max_score = ndcg1
        max_step = step

        print('max_score: ', max_score)
        print('max_step: ', max_step)

    result_save_file = os.path.join(save_score_path, 'test_results_epoch.csv')
    
    if not os.path.exists(result_save_file):
        with open(result_save_file, 'w') as f:
            for para_name in args.para_names:
                f.write(para_name + ',')
            for k in topk:
                f.write('NDCG@{},'.format(k))
            for k in topk:
                f.write('jaccard@{},'.format(k))
            
            f.write('\n')
    # write the results of this epoch to the csv file
    with open(result_save_file, 'a') as f:
        for para_value in args.para_values:
            f.write(str(para_value) + ',')
        if "NDCG" in top_k_scores:
            for ind_k, k in enumerate(topk):
                f.write(str(top_k_scores['NDCG'][ind_k]) + ',')   
        if "jaccard" in top_k_scores:
            for ind_k, k in enumerate(topk):
                f.write(str(top_k_scores['jaccard'][ind_k]) + ',')   
        f.write('\n')
        f.flush()

    with open('{}.json'.format(save_score_path+'/test_results_' + str(step)), 'wt') as f:
        json.dump(generated_dict, f, indent=4)

save_score.write('max_score: '+str(max_score)+'\t'+'max_step: '+str(max_step)+'\n')
all_test_results = pd.read_csv(result_save_file)
best_test_result_under_test = all_test_results
save_folder = os.path.join('topk_scores_jac')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

result_save_epoch = os.path.join('topk_scores_jac', dataset + '_SimpleDyG.csv')
if os.path.exists(result_save_epoch):
    best_test_result_under_test.to_csv(result_save_epoch, mode='a', header=False, index=False)
else:
    best_test_result_under_test.to_csv(result_save_epoch, index=False)

