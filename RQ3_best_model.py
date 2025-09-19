import pandas as pd

def find_best_models(data):
    metrics = ['FDC', 'FBD', 'PLO', 'Ske', 'Kur', 'CL', 'h_max', 'NBC', 'MAPE', 'RD']

    systems = data['System'].unique()
    best_models_dict = {}

    for system in systems:
        system_data = data[data['System'] == system]
        system_data = system_data[system_data['Model'] != 'KNN']
        best_models_dict[system] = {}

        for metric in metrics:
            if len(system_data) == 0: 
                continue
            abs_values = system_data[metric]
            best_model = system_data.loc[abs_values.idxmin(), 'Model']
            best_value = system_data.loc[abs_values.idxmin(), metric]
            best_models_dict[system][metric] = {
                'model': best_model,
                'value': best_value
            }

    return best_models_dict

def generate_system_as_rows_table(best_models_dict, save_path='system_rows_table.tex'):
    
    model_name_mapping = {
        'Support Vector Regression': 'SVR',
        'Linear Regression': 'LR', 
        'Gaussian Process': 'GP',
        'Decision Tree': 'DT',
        'Random Forests': 'RF',
        'DECART': 'DCT',
        'SPLConqueror': 'SPL',
        'DaL': 'DaL',
        'DeepPerf': 'DeP',
        'HINNPerf': 'HIP',
        'kNN': 'kNN',
        'KNN': 'KNN'
    }
    
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
        'storm': 'Storm',
        'hsmgp': 'HSMGP',
        'xgboost': 'XGBoost',
        'hipacc': 'HIPAcc',
        'sql': 'SQL',
        'javagc': 'JavaGC',
        'polly': 'Polly'
    }
    
    system_order = [
        'Apache', '7z', 'DConvert', 'DeepArch', 'ExaStencils',
        'Hadoop', 'MariaDB', 'MongoDB', 'PostgreSQL', 'Redis',
        'Storm', 'HSMGP', 'XGBoost', 'HIPAcc', 'SQL', 'JavaGC', 'Polly'
    ]
    
    metrics = ['FDC', 'FBD', 'Ske', 'Kur', 'PLO', 'CL', 'h_max', 'NBC']
    
    metric_name_mapping = {
        'FDC': 'FDC',
        'FBD': 'FBD', 
        'PLO': 'PLO',
        'Ske': 'Ske',
        'Kur': 'Kur',
        'CL': 'CL',
        'h_max': 'MIE', 
        'NBC': 'NBC'
    }
    
    system_to_display = {system: system_name_mapping.get(system.lower(), system) 
                         for system in best_models_dict.keys()}
    
    available_systems = list(system_to_display.values())
    ordered_systems = [system for system in system_order if system in available_systems]
    for system in available_systems:
        if system not in ordered_systems:
            ordered_systems.append(system)
    
    result_df = pd.DataFrame(index=ordered_systems, columns=metrics)
    values_df = pd.DataFrame(index=ordered_systems, columns=metrics)
    
    for orig_system, metrics_dict in best_models_dict.items():
        system_display = system_to_display[orig_system]
        for metric, info in metrics_dict.items():
            if metric in metrics:  
                model_abbr = model_name_mapping.get(info['model'], info['model'])
                result_df.loc[system_display, metric] = model_abbr
                values_df.loc[system_display, metric] = info['value']
    console_df = result_df.copy()
    console_df.columns = [metric_name_mapping.get(col, col) for col in console_df.columns]
    print(console_df.to_string())
    
    return result_df, values_df

def main():
    data = pd.read_csv('./landscape_data/difference/Summary.csv') 
    data = data[data['Model'] != 'kNN']
    best_models_dict = find_best_models(data)
    generate_system_as_rows_table(best_models_dict, 'system_rows_table.tex')
    
if __name__ == "__main__":
    main()