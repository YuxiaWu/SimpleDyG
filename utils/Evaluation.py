
import math
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_curve, auc
import torch
import os
import json
from tqdm import tqdm
import pandas as pd

class Evaluation:
	def jaccard(self, pred, label):
		pred = set(pred)
		label = set(label)
		return len(pred & label) / len(pred | label)

	def ndcg_k(self,sorted_indices, ground_truth, k):
		dcg, pdcg = 0,0
		for i, item in enumerate(sorted_indices[:k]):
			if item in ground_truth:
				dcg += 1 / math.log(i + 2)
		for i in range(min(len(ground_truth), k)):
			pdcg += 1 / math.log(i + 2)
		return dcg / pdcg
        

def get_eval_metrics(args, model, tokenizer, step, mode = "val"):

    spl_tokens = tokenizer.additional_special_tokens+[tokenizer.bos_token,tokenizer.eos_token,tokenizer.pad_token]
    print('spl_tokens: ', spl_tokens) 
    if mode=='val':
        file_path = args.eval_data_file
        file_path_gt = args.eval_data_gt_file
    elif mode=='test':
        file_path = args.test_data_file
        file_path_gt = args.test_data_gt_file

    with open(file_path, encoding="utf-8") as f:
        data = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    with open(file_path_gt, encoding="utf-8") as f:
        data_gt = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    # make sure the length of data and data_gt are the same
    assert len(data) == len(data_gt)  

    # read the vocab of this timestamp, when evluating, omit the ids not in the vocab
    timestamp = args.timestamp
    dataset = args.dataset

    vocab_file = os.path.join('./vocabs', dataset, timestamp, 'vocab.json')
    vocab = json.load(open(vocab_file, 'r'))

    if args.run_seed:
        save_score_path = os.path.join(args.output_dir, "results_seed_jac", mode+'_score')
    else:
        save_score_path = os.path.join(args.output_dir, "results", mode+'_score')

    if not os.path.exists(save_score_path):
        os.makedirs(save_score_path)

    save_score = open(os.path.join(save_score_path, 'score_all.txt'),'a')

    Eval = Evaluation()   
    model.eval()
    model.to('cuda')

    break_tokens = tokenizer.encode("<|endoftext|>")
    MAX_LEN = model.config.n_ctx
    print('MAX_LEN: ', MAX_LEN)

    generated_dict = {}
    num_data = len(data)

    topk = [5]
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
        # Convert indexed tokens in a PyTorch tensor
        tokens_tensor = torch.tensor([indexed_tokens])

        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to('cuda')
        predicted_index = []
        len_input = len(indexed_tokens)
	gen_len = 0
        while predicted_index not in break_tokens:
            outputs = model(tokens_tensor)
            predictions = outputs[0]
            predicted_index = torch.argmax(predictions[0, -1, :]).item() 
            indexed_tokens += [predicted_index]
            
            predicted_text = tokenizer.decode(indexed_tokens)
            
            tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
            gen_len +=1
            if mode == 'val':
                if gen_len>10:
                    break
            else:
                if len(indexed_tokens) >= MAX_LEN-len(spl_tokens):
                    break

            if tokenizer.decode(indexed_tokens).endswith('<|endoftext|>'):
                break

        predicted_text = tokenizer.decode(indexed_tokens)
        predicted_list = predicted_text.split()
        predicted_list = predicted_list[len_input:]

        # remove all user id in predicted_list,remove the tokens if it belongs to spl_token
        predicted = [token for token in predicted_list if token != user_id]
        predicted = [token for token in predicted if token not in spl_tokens]
        
        # ndcg
        if 'NDCG' in metric_terms:
            for topi, k in enumerate(topk):
                try:
                    result = Eval.ndcg_k(predicted, target_list, k)
                    top_k_scores['NDCG'][topi] += result
                except:
                    print('predicted: ', predicted)
                    print('target_list: ', target_list)
                    top_k_scores['NDCG'][topi] += 0 
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
        generated_dict[i]['NDCG@k'] = str( Eval.ndcg_k(predicted, target_list, 1))
        generated_dict[i]['num_user_test'] = str(num_user_test)

    for metric in metric_terms:
        for topi, k in enumerate(topk):
            # keep 4 digits after decimal point
            top_k_scores[metric][topi] = round(top_k_scores[metric][topi] / num_user_test, 4)
    
    result_save_file = os.path.join(save_score_path, mode+'_results_epoch.csv')

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
        # flush the buffer
        f.flush()
    # save the generated_dict to a json file for each step
    # save_score_path: os.path.join('results', dataset, timestamp, 'test_score')
    with open('{}.json'.format(save_score_path +'/eval_results_' + str(step)), 'wt') as f:
        json.dump(generated_dict, f, indent=4)

    if mode=='test':
        if args.run_seed:
            save_folder = os.path.join('topk_scores_seed_jac')
        else:
            save_folder = os.path.join('topk_scores_finetune')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        result_save_test = os.path.join(save_folder, dataset + '_SimpleDyG.csv')
        test_results = pd.read_csv(result_save_file)

        if os.path.exists(result_save_test):
            test_results.to_csv(result_save_test, mode='a', header=False, index=False)
        else:
            test_results.to_csv(result_save_test, index=False)

    return top_k_scores
