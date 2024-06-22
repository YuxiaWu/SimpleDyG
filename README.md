# SimpleDyG

The code and datasets used for our paper "On the Feasibility of Simple Transformer for Dynamic Graph Modeling" which is accepted by WWW 2024.

# Requirements and Installation


python>=3.9

pytorch>=1.9.1

transformers>=4.24.0

The package can be installed by running the following command.

`pip install -r requirements.txt`

# Benchmark Datasets and Preprocessing (Optional)

You can download the raw datasets and run the preprocessing by yourself. 

Or you can run the training directly using the preprocessed data in `./resources`

## Raw data:

- UCI and  ML-10M: the raw data is the same with  https://github.com/aravindsankar28/DySAT

- Hepth: The dataset can be downloaded from the KDD cup:  https://www.cs.cornell.edu/projects/kddcup/datasets.html

- MMConv: we provide the raw data downloaded from https://github.com/liziliao/MMConv. It is a text-based multi-turn dialog dataset. We preprocess the data by representing the dialog as a graph for each turn based on the annotated attributes. We provide the preprocessed data in `all/data/dialog`

## Let's do preprocessing!  

All the datasets and preprocessing code are in folder `/all_data`. For each dataset, run:

`python preprocess.py ` 


The preprocessed data contains:

- `ml_dataname.csv`: the columns: *u*, *i* is the node Id. *ts* is the time point. *timestamp* is the coarse-grained time steps for temporal alignment.
- `ml_dataname.npy`: the raw link feature. 
- `ml_dataname_node.npy`: the raw node feature. 

Transfer the preprocessed data into sequences for the Transformer model: 

`bash csv2res.sh`

The final data is saved in  `./resources`, including the train/val/test data.

**We provide the final processed data for running our model.**

# Train the model 

`bash train_UCI_13.sh`

During training, the following output files are generated:

- `./tokenizers`: the tokenizers for each timestamp

- `./vocabs`: the vocab for each timestamp

- `./output`: the saved checkpoint of the model

- `./results`: the output results and metrics 

- `./runs`: the tensorboard results


# Evaluation 

`bash eval_single_step.py`

# Citation
```
@inproceedings{wu2024feasibility,
  title={On the Feasibility of Simple Transformer for Dynamic Graph Modeling},
  author={Wu, Yuxia and Fang, Yuan and Liao, Lizi},
  booktitle={Proceedings of the ACM on Web Conference 2024},
  pages={870--880},
  year={2024}
}

```
