# GCNDFMDA
GCNDFMDA: Predict miRNA-disease associations using a variant of deep forest model and improving feature vectors by graph convolutional network

## Requirements
  * xgboost==2.0.3
  * matplotlib==0.1.6
  * pandas==1.5.3
  * joblib==1.2.0
  * torch==2.0.0+cu118
  * pandas==1.5.3
  * scipy==1.9.3
  * scikit-learn==1.2.2
  * numpy==1.22.4
  * torch_geometric==2.3.1
  * xlrd==1.2.0
  * openpyxl==3.1.4


## Usage
  * Open ```params.py``` and choose ```dataset```, ```type_eval``` and ```nloop```, respectively.  ```number of repeat times```
    (- dataset: "HMDD v2.0" / "HMDD v3.2" / "INDE_TEST", respectively to HMDD v2.0, HMDD v3.2 and Independent dataset.
     - type_eval = "KFOLD"  / "DIS_K" / "DENO_MI", respectively to 5-fold-CV, Case studies and Denovo miRNAs.
     - nloop = ```number of repeat times```
     - Defaut: 5fold-CV in HMDD v2.0 dataset)
  * Unzip OUT.rar
  
