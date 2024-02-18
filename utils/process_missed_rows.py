import pandas as pd
import numpy as np
import argparse
from typing import Union
from tabulate import tabulate

# Defining script parsing
parser = argparse.ArgumentParser(description='Process missing rows.')
parser.add_argument("process_filepath", "-data_folder")
parser.add_argument("curr_folder", "-curr_folder")
parser.add_argument("add_headers", "-h")

# Defining args
args = parser.parse_args()
data_folder = args.data_folder
curr_folder = args.label_filepcurr_folderath
add_headers = args.add_headers

# Helper Function
def process_missed(
    df: pd.DataFrame,
    missed_pos:Union[list, np.ndarray],
    data_folder,
    curr_folder
):
    df_missed = pd.DataFrame(columns=df.columns)
    # df_missed.columns
    algo_name_list = ['mutect2', 'varscan', 'vardict', 'freebayes']
    sample_name = df['Sample_Name'].unique()[0]
    for chr_pos in missed_pos:
        print(f"Processing {chr_pos}")
        chr = chr_pos.split('_')[0]
        pos = chr_pos.split('_')[1]

        m2_ref = None
        m2_alt = None
        f_ref = None
        f_alt = None
        vd_ref = None
        vd_alt = None
        vs_ref = None
        vs_alt = None

        curr_row = {}
        curr_row['Chr'] = chr
        curr_row['START_POS_REF'] = pos
        curr_row['END_POS_REF'] = pos
        curr_row['FILTER_Mutect2'] = False
        curr_row['FILTER_Freebayes'] = False
        curr_row['FILTER_Vardict'] = False
        curr_row['FILTER_Varscan'] = False

        curr_row['Sample_Name'] = sample_name
        for algo_name in algo_name_list:
            print(algo_name)
            if algo_name == 'mutect2':
                algo_features = m2_cols
                prefix = 'm2_'
            elif algo_name == 'varscan':
                algo_features = vs_cols
                prefix = 'vs_'
            elif  algo_name == 'vardict':
                algo_features = vd_cols
                prefix = 'vd_'
            elif algo_name == 'freebayes':
                algo_features = f_cols
                prefix = 'f_'
            with open('{}/{}/{}-{}.vcf'.format(data_folder, curr_folder, curr_folder, algo_name), 'r') as f:
                lines = [line.rstrip() for line in f if line.startswith(f'{chr}\t{pos}\t')]
                if len(lines) == 1:
                    m2_ref = 'NA'
                    m2_alt = 'NA'
                    f_ref = 'NA'
                    f_alt = 'NA'
                    vd_ref = 'NA'
                    vd_alt = 'NA'
                    vs_ref = 'NA'
                    vs_alt = 'NA'

                    line = lines[0]
                    cols = line.split('\t')
                    if algo_name == 'mutect2':
                        m2_ref = cols[3]
                        m2_alt = cols[4]
                    elif algo_name == 'varscan':
                        vs_ref = cols[3]
                        vs_alt = cols[4]
                    elif  algo_name == 'vardict':
                        vd_ref = cols[3]
                        vd_alt = cols[4]
                    elif algo_name == 'freebayes':
                        f_ref = cols[3]
                        f_alt = cols[4]
                    
                    info = cols[7]
                    feature_list = info.split(';')
                    for feature in feature_list:

                        if len(feature.split('=')) == 1:
                            feature_name = feature.split('=')[0]
                            feature_val = None
                        else:
                            feature_name = feature.split('=')[0]
                            feature_val = feature.split('=')[1]
                        if feature_name in algo_features:
                            curr_row[prefix + feature_name] = feature_val
                elif len(lines) > 1:
                    input("Unexpected: algo contains more than 1 entry for 1 position")
                else:
                    for feature_name in algo_features:
                        curr_row[prefix + feature_name] = None
        
        # None of the 4 algos pick up this
        if m2_ref is None:
            continue

        for i in [m2_ref, f_ref, vd_ref, vs_ref]:
            if i != 'NA':
                curr_row['REF'] = i
                break
        for i in [m2_alt, f_alt, vd_alt, vs_alt]:
            if i != 'NA':
                curr_row['ALT'] = i
                break
        curr_row['REF_MFVdVs'] = f'{m2_ref}/{f_ref}/{vd_ref}/{vs_ref}/'
        curr_row['ALT_MFVdVs'] = f'{m2_alt}/{f_alt}/{vd_alt}/{vs_alt}/'
        curr_row['_merge'] = 'right_only'

        for col in df_missed.columns:
            if col not in curr_row.keys():
                print(f'missed col: {col}')
                curr_row[col] = None
            
        df_missed = df_missed._append(curr_row, ignore_index = True) 
        print(tabulate(df_missed, headers='keys', tablefmt='psql'))

    return df_missed


# Assert that filetype is text
try:
    process_filepath = f'{data_folder}/snv-parse-{curr_folder}_all_features.txt'
    label_filepath = f'{data_folder}/{curr_folder}/{curr_folder}_truth.bed'

    filetype = process_filepath.split("/")[-1].split(".")
    assert filetype == "txt", "Wrong filetype"
    # Processing if headers exist
    if add_headers == "true":
        names=['Chr', 'START_POS_REF', 'END_POS_REF']
    else:
        names=None      

    df1 = pd.read_csv(
        f'{process_filepath}', delimiter='\t'
        )
    df_label = pd.read_csv(
        f'{label_filepath}_truth.bed', delimiter='\t',names=names
        )

    df1['Chr'] = df1['Chr'].astype(str)
    df_label['Chr'] = df_label['Chr'].astype(str)
    df = df1.merge(
        df_label,
        on=['Chr', 'START_POS_REF', 'END_POS_REF'],
        how='outer',
        indicator=True
        )

    df[df['_merge'] == 'right_only']
    m2_cols = [col.split('_', 1)[1] for col in df.columns if col.startswith('m2_')]
    f_cols = [col.split('_', 1)[1] for col in df.columns if col.startswith('f_')]
    vs_cols = [col.split('_', 1)[1] for col in df.columns if col.startswith('vs_')]
    vd_cols = [col.split('_', 1)[1] for col in df.columns if col.startswith('vd_')]

    # Processing joined files
    df_right_only = df[df['_merge'] == 'right_only']
    df_right_only.iloc[:,'Missed_pos'] = df_right_only['Chr'].astype(str) + '_' + df_right_only['START_POS_REF'].astype(str)
    missed_pos = df_right_only['Missed_pos'].unique()
    df_right_only.drop(['Missed_pos'], axis=1, inplace=True)

    # Processing Missing
    df_missed = process_missed(
        df_joined,
        missed_pos,
        data_folder,
        curr_folder
    )

    # Processing Type 
    df_missed['START_POS_REF'] = df_missed['START_POS_REF'].astype(int)
    df_missed['END_POS_REF'] = df_missed['END_POS_REF'].astype(int)
    df_missed['Chr'] = df_missed['Chr'].astype(str) 
    print(f" INFO on missed dataframe :: {df_missed}")

    # To get entire Dataframe
    df_all = pd.concat([df1, df_missed], ignore_index=True)
    df_all_joined = df_all.merge(df_label, on=['Chr', 'START_POS_REF', 'END_POS_REF'], how='outer', indicator=True)
    df_all_joined['is_snv'] = np.where((df_all_joined['_merge'] == 'both'), True, False)
    df_all_joined.drop(['_merge'], axis=1, inplace=True)
    df_all_joined.to_csv(f'{curr_folder}_final.csv', sep='\t')


except Exception as e:
    print(f" ERRROR :: [{e}]")