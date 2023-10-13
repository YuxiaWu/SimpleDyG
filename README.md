

##  Requirements:

python=3.9

pytorch=1.9.1

transformers=4.24.0

The package can be installed by running the following command.

`pip install -r requirements.txt`

## Benchmark Datasets and Preprocessing

# Four datasets:

UCI and  ML-10M: the raw-data is the same with  https://github.com/aravindsankar28/DySAT

Hepth: The dataset can be download from the kddcup:  https://www.cs.cornell.edu/projects/kddcup/datasets.html

MMConv: we provide the raw data is downloaded from https://github.com/liziliao/MMConv. It is a text-based multi-turn dialog dataset. We preprocess the data by representing the dialog as a graph for each turn based on the annotated attributes. We provide the preprocessed data in "all/data/dialog"

All the datasets and preprocessing code is in folder "/all_data". For each dataset, run:

`python preprocess.py ` 


Transfer the preprocessed data into sequences for Transformer model: 

`bash csv2res.sh`

The final data is saved in:  ./resources. including the train/val/test data.

We provide the final processed data of UCI for runing our model.

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

