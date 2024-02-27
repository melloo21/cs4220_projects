# cs4220_projects
Repository for Project 1 - Mutation calling in cancer

### Installation
```
git clone 
cd 
pip install -r requirements.txt
```

### Scripts
* eda_preprocess/get_missed_rows.ipynb: iterate all rows in .bed file that are skipped by parse-vcf-snv-wrapper.R, extract features and add to parsed output if the row is called by at least one of M2, Freebayes, Vd, Vs regardless of each algorithm's PASS/REJECT label. Needed to increase the number of True labels in training set
* eda_preprocess/eda.ipynb: load and visualize data
* feature_engineering/make_dataset.ipynb: Generates the dataset shown in /cs4220_projects/data/feature_set, this will be the final x feature list and y label list. It includes the downsampling , upsampling methods as well.
* feature_engineering/combined_dataset.ipynb: Generates combined dataset 
* utils: includes all the utility functions required to model and evaluate the model
* model_selection: provides methodology to select the best model
* training: file to train the models via hyperparameter tuning
* model_evaluation.ipynb : includes evaluation of models and SHAP analysis

### Folders
* /cs4220_projects/data/: contains all datasets use for eda_preprocess/ feature_engineering/ model
* /cs4220_projects/model_assets/: contains all model weights in joblib file
* /cs4220_projects/predictions/: contains all past bed prediction positions
* /cs4220_projects/plots/: saved distribution plots

