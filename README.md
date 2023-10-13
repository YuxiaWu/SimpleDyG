

##  Requirements:
python 3.9
pytorch 1.9.1


## Benchmark Datasets and Preprocessing

All the datasets and preprocessing code is in folder "/all_data". For each dataset:

`python preprocess.py ` 


Transfer the preprocessed data into sequences for Transformer model: 

`bash csv2res.sh`

The final data is saved in:  ./resources. including the train/val/test data

## Train the model 

`bash train_UCI_13.sh`

Will obtain the following output files:

./tokenizers: the tokenizers for each timestamp
./vocabs: the vocab for each timestamp
./output: the checkpoint 
./results: the output results and metrics 
./runs: the tensorboard results


## Evaluation 

`bash eval_single_step.py`
