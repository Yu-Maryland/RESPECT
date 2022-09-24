# RESPECT

Project related to the paper rejected by ICLR 2022:

Model path: models/scheduling_level_aligned/

dataset path: models/scheduling_level_aligned/

Code path: 
#LSTM-LSTM
/mnt/raid10/qiwei/Topological_Sorting/Topological_Sorting_LSTM_Version/ 
#LSTM-Transformer
/mnt/raid10/qiwei/Topological_Sorting/Topological_Sorting_LSTM_Transformer_Version/
#Transformer-LSTM
/mnt/raid10/qiwei/Topological_Sorting/Topological_Sorting_Transformer_LSTM_Version/
#Transformer-Transformer
/mnt/raid10/qiwei/Topological_Sorting/Topological_Sorting_Transformer_Version/

Encoder-decoder pairs differers in each folder above. Basic setting is same.

Take #LSTM-LSTM as example:
Command line setting parameters can be found in options.py.
To run the system, two parameters must be set in the command line or in options.py manually:”--train_dataset_path”, “eval_dataset_path”
Bash command:
python run.py --train_dataset_path TDataset --eval_dataset_path EDataset 

Construction of TDataset and EDataset:
Code Path:
/mnt/raid10/qiwei/Topological_Sorting/dataset/
Command:
python dataset_generator.py