# cs4220_projects
Repository for Project 1 - Mutation calling in cancer

### Installation
```
git clone 
cd 
pip install -r requirements.txt
```

### Scripts
* eda_preprocess/get_missed_rows.ipynb: iterate all rows in .bed file that are skipped by parse-vcf-snv-wrapper.R, extract features and add to parsed output if the row is called by at least one of M2, Freebayes, Vd, Vs regardless of each algorithm's PASS/REJECT label.
* eda_preprocess/eda.ipynb: load and visualize data
