# ChemTS
##  Requirements 
1. [Python](https://www.anaconda.com/download/)>=2.7 
2. [Keras](https://github.com/fchollet/keras) (version 2.0.5) If you installed the newest version of keras, some errors will show up. Please change it back to keras 2.0.5 by pip install keras==2.0.5. 
3. [rdkit](https://anaconda.org/rdkit/rdkit)

##  Train a RNN model for molecule generation
1. cd train_RNN
2. Run python train_RNN.py to train the RNN model. GPU is highly recommended for reducing the training time.

##  MCTS for logP optimization
1. cd mcts_logp_improved_version
2. Run python mcts_logp.py
