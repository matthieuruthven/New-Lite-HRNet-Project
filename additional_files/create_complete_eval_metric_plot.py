# create_complete_eval_metric_plot.py
# Code to create plots of evaluation metrics

# Author: Matthieu Ruthven (matthieu.ruthven@uni.lu)
# Last modified: 16th October 2023

# Import required modules
from pathlib import Path
import pandas as pd
from os import scandir
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

# # Path to root directory
# root_dir_path = Path('/data/mruthven/lite-hrnet-dsnt/work_dirs_old')

# # List of folders containing results of interest
# dir_name_list = ['td-hm_litehrnet-18_speedplus-640x640_lbx_to_lbx',
#                  'td-hm_litehrnet-18_speedplus-640x640_slp_to_slp',
#                  'td-hm_litehrnet-18_speedplus-640x640_syn_to_syn',
#                  'td-hm_litehrnet-30_speedplus-640x640_lbx_to_lbx',
#                  'td-hm_litehrnet-30_speedplus-640x640_slp_to_slp']

# Path to root directory
root_dir_path = Path('/data/mruthven/lite-hrnet-dsnt/work_dirs')

# List of folders containing results of interest
dir_name_list = ['td-hm_litehrnet-18_speedplus-640x640_lbx_to_lbx',
                 'td-hm_litehrnet-18_speedplus-640x640_slp_to_slp',
                 'td-hm_litehrnet-18_speedplus-640x640_syn_to_syn',
                 'td-hm_litehrnet-30_speedplus-640x640_lbx_to_lbx',
                 'td-hm_litehrnet-30_speedplus-640x640_slp_to_slp',
                 'td-hm_litehrnet-30_speedplus-640x640_syn_to_syn',
                 'td-hm_litehrnet-18_speedplus-640x640_lbx_to_lbx_pretrained',
                 'td-hm_litehrnet-18_speedplus-640x640_slp_to_slp_pretrained',
                 'td-hm_litehrnet-30_speedplus-640x640_lbx_to_lbx_pretrained',
                 'td-hm_litehrnet-30_speedplus-640x640_slp_to_slp_pretrained']

# Preallocate pandas DataFrame for resuts
all_df = pd.DataFrame()

# For each folder
for dir_name in dir_name_list:

    # Identify Lite-HRNet architecture
    arch = int(dir_name[16:18])

    # Identify source and target domains
    src_dom = dir_name.split('_')
    tgt_dom = src_dom[5]
    src_dom = src_dom[3]

    # List of folders in folder
    subdir_name = [f.name for f in scandir(root_dir_path / dir_name) if f.is_dir()]
    
    # Check there is only a single folder
    assert len(subdir_name) == 1, f'There are more than one subfolders in {dir_name}'
    subdir_name = subdir_name[0]

    # Path to JSON file containing results
    json_path = root_dir_path / dir_name / subdir_name / 'vis_data' / f'{subdir_name}.json'

    # Read JSON file
    df = pd.read_json(json_path, lines=True)

    # Flag
    no_val = False

    # If there are validation results
    if 'PCK' in df.columns:

        # Extract results on validation dataset
        tmp_df = df.dropna(subset='PCK')

        # Extract relevant columns
        tmp_df = tmp_df[['step', 'PCK']]

        # Rename columns
        tmp_df.rename(columns={'step': 'Epoch',
                               'PCK': 'Mean Proportion of Correct Keypoints'}, inplace=True)

        # Add columns
        tmp_df['Lite-HRNet Version'] = arch
        tmp_df['Source/Target'] = f'{src_dom.upper()}/{tgt_dom.upper()}'
        tmp_df['Split'] = 'Validation'
        if 'pretrained' in dir_name:
            tmp_df['Pretraining'] = True
        else:
            tmp_df['Pretraining'] = False

        # Update all_df
        all_df = pd.concat([all_df, tmp_df])

        # Extract results on training dataset
        tmp_df = df[df['PCK'].isna()]

    else:
        
        # Extract results on training dataset
        tmp_df = df

        # Update flag
        no_val = True

    # Extract relevant columns
    tmp_df = tmp_df[['epoch', 'acc_pose']]

    # Calculate mean accuracy per epoch
    tmp_df = tmp_df.groupby('epoch').mean().reset_index()

    # Calculate position of iteration within its epoch
    # tmp_df['position_within_epoch'] = tmp_df.groupby('epoch').cumcount() + 1

    # Calculate number of iterations in each epoch
    # tmp_df['iterations_per_epoch'] = tmp_df.groupby('epoch')['epoch'].transform('size')

    # Calculate a scalar
    # tmp_df['Epoch'] = tmp_df['epoch'] + (tmp_df['position_within_epoch'] - 1) / tmp_df['iterations_per_epoch']

    # Extract relevant columns
    # tmp_df = tmp_df[['Epoch', 'acc_pose']]

    # Rename column
    # tmp_df.rename(columns={'acc_pose': 'Mean Proportion of Correct Keypoints'}, inplace=True)

    # Rename columns
    tmp_df.rename(columns={'epoch': 'Epoch',
                           'acc_pose': 'Mean Proportion of Correct Keypoints'}, inplace=True)
    
    # Add columns
    tmp_df['Lite-HRNet Version'] = arch
    tmp_df['Source/Target'] = f'{src_dom.upper()}/{tgt_dom.upper()}'
    tmp_df['Split'] = 'Training'
    if 'pretrained' in dir_name:
        tmp_df['Pretraining'] = True
    else:
        tmp_df['Pretraining'] = False

    # Update all_df
    all_df = pd.concat([all_df, tmp_df])
        
# Set seaborn style
sns.set_style('darkgrid')

# If there are no validation results
if no_val:

    # Create seaborn plot
    g = sns.lineplot(data=all_df,
                    x='Epoch',
                    y='Mean Proportion of Correct Keypoints',
                    hue='Source/Target',
                    style='Lite-HRNet Version')
    
    # Save plot
    plt.savefig(f'New-Lite-HRNet_Train_PCK_Plot_{datetime.date.today()}.png', bbox_inches='tight')
    
else:

    # Create seaborn plot
    g = sns.relplot(data=all_df,
                    kind='line',
                    x='Epoch',
                    y='Mean Proportion of Correct Keypoints',
                    hue='Lite-HRNet Version',
                    col='Source/Target',
                    row='Pretraining',
                    style='Split',
                    style_order=['Training', 'Validation'])
    
    # Save plot
    plt.savefig(f'New-Lite-HRNet_Train_Val_PCK_Plot_{datetime.date.today()}.png', bbox_inches='tight')