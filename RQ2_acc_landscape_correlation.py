import warnings
warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
import pandas as pd
import os
import numpy as np
import math

data_set_names = ['Apache', '7z','dconvert','deeparch',
                  'exastencils','Hadoop','MariaDB',"MongoDB","PostgreSQL",
                  'redis', 'spark', 'storm',  'hsmgp', 
                  'xgboost', 'hipacc', 'SQL', 'javagc','polly'] 

metrics_order = ['FDC', 'FBD', 'Ske', 'Kur', 'PLO', 'CL', 'h_max', 'NBC']  
global_metrics = ['FDC', 'FBD', 'Ske', 'Kur']  
direct_metrics = ['PLO', 'CL', 'h_max', 'NBC'] 
reverse_metrics = ['CL']  

correlation_results_by_set = {name: {} for name in data_set_names}

for index, data_set_name in enumerate(data_set_names):
    model_directory_path = f"./landscape_data/model/{data_set_name}"
    model_files = [f for f in os.listdir(model_directory_path) if f.endswith('.csv')]
    
    real_directory_path = f"./landscape_data/real/{data_set_name}"
    real_file_path = os.path.join(real_directory_path, "DT.csv")
    
    if not os.path.exists(real_file_path):
        print(f"Warning: Real data file not found for {data_set_name}")
        continue
    
    combined_model_df = pd.read_csv(os.path.join(model_directory_path, model_files[0]))
    for file in model_files[1:]:
        df = pd.read_csv(os.path.join(model_directory_path, file))
        combined_model_df = pd.concat([combined_model_df, df], ignore_index=True)
    
    real_df = pd.read_csv(real_file_path)
    real_row = real_df.iloc[0]  
    
    columns_last = combined_model_df.columns[-2:]

    correlation_results = {}

    for col_mid in metrics_order:
        if col_mid not in combined_model_df.columns:
            print(f"Warning: Column {col_mid} not found in model data for {data_set_name}")
            continue
            
        for col_last in columns_last:
            if col_mid not in real_df.columns or col_last not in real_df.columns:
                print(f"Warning: Column {col_mid} or {col_last} not found in real data for {data_set_name}")
                continue
                
            model_mid = combined_model_df[col_mid].astype(float)
            model_last = combined_model_df[col_last].astype(float)
            real_mid_value = float(real_row[col_mid])  
            real_last_value = float(real_row[col_last]) 
            
            if col_mid in global_metrics:
                abs_diff_mid = np.abs(model_mid - real_mid_value)
                abs_diff_last = np.abs(model_last - real_last_value)
                data1, data2 = abs_diff_mid, abs_diff_last
            else:
                data1, data2 = model_mid, model_last
            
            try:
                new_list1 = []
                new_list2 = []

                for i in range(len(data1)):
                    if not math.isnan(data1.iloc[i]) and not math.isnan(data2.iloc[i]):
                        new_list1.append(data1.iloc[i])
                        new_list2.append(data2.iloc[i])
                
                if len(new_list1) > 1:
                    correlation, p_value = spearmanr(new_list1, new_list2)
                    
                    if col_mid in reverse_metrics:
                        correlation = -correlation
                    
                    if p_value < 1.1:
                        correlation, p_value = round(correlation,4), round(p_value,4)
                    else:
                        correlation, p_value = '-','-'
                else:
                    correlation, p_value = '-', '-'
                    
                correlation_results[(col_mid, col_last)] = (correlation, p_value)
            except Exception as e:
                print(f"Error processing {col_mid} vs {col_last} for {data_set_name}: {e}")
                correlation_results[(col_mid, col_last)] = ("Nan", "Nan")

    print(str(data_set_name) + str(list(correlation_results.values())))