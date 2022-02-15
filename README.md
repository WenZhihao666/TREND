# TREND: TempoRal Event and Node Dynamics for Graph Representation Learning
We provide the implementaion of TREND model, which is the source code for the WWW 2022 paper
"TREND: TempoRal Event and Node Dynamics for Graph Representation Learning". 

The repository is organised as follows:
- dataset/: the directory of data sets, and it contains the cit-HepTh data set as the example. 
- res/: the directory of saved models.
- Emlp.py: the transfer function for Hawkes process.
- data_dyn_cite.py: training data preprocessing.
- data_tlp_cite.py: testing data preperation.
- dgnn.py: the Hawkes process based GNN.
- film.py: the event-conditioned transformation.
- main_test: the testing entrance.
- main_train: the training entrance.
- model: the whole model of proposed TREND.
- node_relu: the MLP of node-dynamics predictor.


## Requirements

  To install requirements:

    pip install -r requirements.txt

## Train and test

  To train the model in the paper:
  
    python main_train.py
    
  To test the trained model:
  
    python main_test.py
    
## Cite
	@inproceedings{wen2022trend,
		title = {TREND: TempoRal Event and Node Dynamics for Graph Representation Learning},
		author = {Wen, Zhihao and Fang, Yuan},
		booktitle = {Proceedings of the Web Conference 2022},
		year = {2022}
	}

