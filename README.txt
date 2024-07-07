
Running Tensor.py

python tensor.py [FILENAME] [DENSITY] [PRECISION]
Example

CSV PARSER
python tensor.py xyz.csv 64 2

NPY PARSER

Generate Dataset:
Testing param: python tensor.py qm9_filtered.npy 64 2
Scientific param: python tensor.py qm9_filtered.npy 729 4

python tensor.py [filename] [output_file] [DENSITY] [PRECISION]

Model Training:

python torch_model.py
python mlp_model.py
python cnn_model.py


'''
Training format:

index,formula,tensor,rotation0,rotation1,rotation2
079782,O C C C O C C N C H H H H H H H,0.0 0.0 0.0 0.05 0.0 0.01 0.08 0.0 0.0 0.36 1.49 0.0 0.0 0.05 0.13 0.0 0.0 0.0 0.59 2.0 1.02 2.0 2.37 3.63 0.21 2.18 8.92 0.01 0.0 0.0 3.2 4.77 0.0 2.68 6.42 0.9 1.1099999999999999 5.35 2.74 3.25 0.63 3.46 0.01 7.38 0.02 0.12 3.2199999999999998 4.79 0.0 0.71 1.59 0.0 0.17 0.22 7.02 0.8300000000000001 1.72 8.290000000000001 3.4 2.87 1.97 6.159999999999999 0.01 0.0,-0.0,-0.0,-0.0
005049,N C N C C N C N H H H H,0.04 0.0 0.0 0.0 1.3 1.38 0.0 0.0 0.45999999999999996 0.74 0.0 0.0 0.15 0.02 0.0 0.0 0.0 0.0 0.82 1.23 1.23 4.99 1.46 2.06 8.690000000000001 3.57 0.03 0.0 4.03 0.85 0.19 0.64 0.0 0.0 1.67 2.36 0.0 0.0 3.15 4.43 0.13 1.29 0.49 0.0 0.09 6.9799999999999995 2.95 3.5799999999999996 0.0 0.0 1.22 1.71 0.84 4.51 2.49 3.49 7.11 4.890000000000001 0.0 0.0 4.51 1.11 0.0 0.01,-14.99,-12.5,-4979.91

'''


#how to visulaize nn
PyTorchViz
visualization package. 38

Plan of Execution:

6 properties, 6 models

Chiral Center Existence
01 Chiral Center
Number Chiral Centers
RS classification 1 Chiral
posneg Classification 1 Chiral 
posneg all Classification all Chiral


tensor_dataset_v2.csv 
index, tensor, chiral0, rotation0

RF model on each, migrate to Bluehive, test CNN. 

python [model_name].py [dataset] [output_file] [test_size] [hidden_layer_sizes] [max_iter] 

python chiral_model.py tensor_dataset_v2.csv chiral_model_pred_v2.csv .05 10 100

#Recommended Traing Hyperparamters
python chiral_model.py tensor_dataset_v2.csv chiral_model_pred_v2.csv .2 100 2000
python chiral_01_model.py tensor_dataset_v2.csv chiral_model_pred_v2.csv .2 100 2000

python number_chiral_classifier.py tensor_dataset_v3.csv number_chiral_classifier_pred_v3.csv .2 100 2000

python rs_model.py tensor_dataset_v2.csv rs_model_pred_v2.csv .2 100 2000


python posneg_model.py tensor_dataset_v2.csv posneg_model_pred_v2.csv .2 100 2000

python posneg_all_model.py tensor_dataset_v2.csv chiral_model_pred_v2.csv .2 100 2000






Predictions o

CITATIONS:

https://irvinelab.uchicago.edu/papers/EI12.pdf
https://www.biorxiv.org/content/10.1101/2022.08.24.505155v1.full.pdf

https://www.nature.com/articles/s41377-020-00367-8

https://www.condmatjclub.org/uploads/2013/06/JCCM_JUNE_2013_03.pdf

https://arxiv.org/pdf/2008.01715
