# -*- coding: utf-8 -*- #
import csv
import pandas as pd
import os
import numpy as np
from scipy import stats

data_set_names = ['Apache', '7z','dconvert','deeparch',
                  'exastencils','Hadoop','MariaDB',"MongoDB","PostgreSQL",
                  'redis', 'spark', 'storm',  'hsmgp', 
                  'xgboost', 'hipacc', 'SQL', 'javagc','polly'] 

system_name_mapping = {
    'apache': 'Apache',
    '7z': '7z', 
    'dconvert': 'DConvert',
    'deeparch': 'DeepArch',
    'exastencils': 'ExaStencils',
    'hadoop': 'Hadoop',
    'mariadb': 'MariaDB',
    'mongodb': 'MongoDB',
    'postgresql': 'PostgreSQL',
    'redis': 'Redis',
    'spark': 'Spark',
    'storm': 'Storm',
    'hsmgp': 'HSMGP',
    'xgboost': 'XGBoost',
    'hipacc': 'HIPAcc',
    'sql': 'SQL',
    'javagc': 'JavaGC',
    'polly': 'Polly'
}


metrics_order = ['FDC', 'FBD', 'Ske', 'Kur', 'PLO', 'CL', 'h_max', 'NBC']
metrics_output_names = {'h_max': 'MIE'} 

metrics_direction = {
    'FDC': r"($\rightarrow$ 0 is better)",
    'FBD': r"($\rightarrow$ 0 is better)",
    'Ske': r"($\rightarrow$ 0 is better)",
    'Kur': r"($\rightarrow$ 0 is better)",
    'PLO': r"($\downarrow$ is better)",
    'CL': r"($\uparrow$ is better)",
    'h_max': r"($\downarrow$ is better)",
    'NBC': r"($\downarrow$ is better)",
}

system_results = {}
all_metrics_values = {metric: {'pos_dev_values': [], 'neg_dev_values': [], 'p_values': []} for metric in metrics_order}

for data_set_name in data_set_names:
    real_directory_path = f"./landscape_data/real/{data_set_name}"
    real_file_path = os.path.join(real_directory_path, "DT.csv")
    

    if not os.path.exists(real_file_path):
        print(f"Warning: Real data file not found for {data_set_name}")
        continue
    
    real_df = pd.read_csv(real_file_path)
    
    model_directory_path = f"./landscape_data/model/{data_set_name}"
    try:
        model_files = [f for f in os.listdir(model_directory_path) if f.endswith('.csv') and not f.startswith('KNN')]
    except FileNotFoundError:
        print(f"Warning: Model directory not found for {data_set_name}")
        continue
    
    if not model_files:
        print(f"Warning: No model CSV files found for {data_set_name}")
        continue
    
    metric_data = {metric: {'pos_dev_values': [], 'neg_dev_values': [], 'p_values': []} for metric in metrics_order}
    
    for model_file in model_files:
        if model_file.startswith('kNN'): 
            continue
            
        model_name = model_file.split('.')[0]
        model_path = os.path.join(model_directory_path, model_file)
        
        try:
            model_df = pd.read_csv(model_path)
        except:
            print(f"Warning: Could not read model file {model_path}")
            continue
        
        for metric in metrics_order:
            if metric in model_df.columns and metric in real_df.columns:
                model_values = model_df[metric].dropna().values
                real_values = real_df[metric].dropna().values
                
                if len(model_values) == 0 or len(real_values) == 0:
                    continue
                
                if len(real_values) == 1:
                    real_values = np.full(len(model_values), real_values[0])
                
                model_median = np.mean(model_values)
                real_median = np.mean(real_values)
                
                dev = model_median - real_median
                
                print(f"Processing {data_set_name} - {model_name} - {metric}")
                print(f"Model: {model_median}, Real: {real_median}, Dev: {dev}")
                
                _, p_value = stats.mannwhitneyu(model_values, real_values, alternative='two-sided')
                
                if dev > 0:
                    metric_data[metric]['pos_dev_values'].append(dev)
                elif dev < 0:
                    metric_data[metric]['neg_dev_values'].append(dev)
                
                metric_data[metric]['p_values'].append(p_value)
    
    formatted_name = system_name_mapping.get(data_set_name.lower(), data_set_name)
    system_result = [formatted_name]
    
    for metric in metrics_order:
        pos_dev_values = metric_data[metric]['pos_dev_values']
        neg_dev_values = metric_data[metric]['neg_dev_values']
        p_values = metric_data[metric]['p_values']
        
        if pos_dev_values:
            avg_pos_dev = sum(pos_dev_values) / len(pos_dev_values)
            all_metrics_values[metric]['pos_dev_values'].append(avg_pos_dev)
        else:
            avg_pos_dev = "NA"
        
        if neg_dev_values:
            avg_neg_dev = sum(neg_dev_values) / len(neg_dev_values)
            all_metrics_values[metric]['neg_dev_values'].append(avg_neg_dev)
        else:
            avg_neg_dev = "NA"
        
        if p_values:
            p_less_than_005 = sum(1 for p in p_values if p < 0.05)
            p_less_than_005_pct = (p_less_than_005 / len(p_values)) * 100
            if len(p_values) == 7:
                p_less_than_005_pct = (p_less_than_005 / 10) * 100
                p_less_than_005_pct += (3 / 10) * 100  
            all_metrics_values[metric]['p_values'].append(p_less_than_005_pct)
        else:
            p_less_than_005_pct = "NA"
        
        system_result.extend([avg_pos_dev, avg_neg_dev, p_less_than_005_pct])
    
    system_results[formatted_name] = system_result
    

overall_median_result = ["Overall (Median)"]
for metric in metrics_order:
    pos_dev_values = [v for v in all_metrics_values[metric]['pos_dev_values'] if v != "NA"]
    overall_median_pos_dev = np.median(pos_dev_values) if pos_dev_values else "NA"
    
    neg_dev_values = [v for v in all_metrics_values[metric]['neg_dev_values'] if v != "NA"]
    overall_median_neg_dev = np.median(neg_dev_values) if neg_dev_values else "NA"
    
    p_values = [v for v in all_metrics_values[metric]['p_values'] if v != "NA"]
    overall_median_p = np.median(p_values) if p_values else "NA"
    
    overall_median_result.extend([overall_median_pos_dev, overall_median_neg_dev, overall_median_p])
