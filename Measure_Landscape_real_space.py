# -*- coding: utf-8 -*-
import pandas as pd
import os
import csv
import numpy as np

from numpy import genfromtxt
from utils.general import get_non_zero_indexes
from pflacco.classical_ela_features import *
from Landscape_utils import *


def run_main(whole_data, found_file, name, lower_bound, upper_bound, real_best, unique_elements_per_column, min_):
    # Calculate accuracy metrics MAPE and RD
    map_real = {tuple(whole_data[n][:-1]): float(whole_data[n][-1]) for n in range(len(whole_data))}
    # Start converting to format for API use [df: configs ===== last_column_values: values]
    last_column_values = pd.Series(list(map_real.values()))
    numpy_array = np.array(list(map_real.keys()))
    # Convert numpy array to pandas DataFrame
    df = pd.DataFrame(numpy_array)
    # Set the first column as index column
    df.index = pd.RangeIndex(start=0, stop=len(df), step=1)
    landscape1 = landscape(map=map_real, Min=min_, df=df, last_column_values=last_column_values, lower_bound=lower_bound, upper_bound=upper_bound)
    FDC = landscape1.calculate_FDC()
    print("FDC:", FDC)
    FBD = landscape1.calculate_best_distance(real_bests=real_best)
    print("From best distance: ", FBD)  # Requires global optimum
    PLO = landscape1.calculate_Proportion_of_local_optimal(unique_elements_per_column)
    print("PLO: ", PLO)
    Ske = landscape1.calculate_skewness()
    print("skewness: ", Ske)
    Kur = landscape1.calculate_kurtosis()
    print("kurtosis: ", Kur)
    CL = landscape1.calculate_correlation_length()
    print("Correlation length:", CL)
    MIE = landscape1.calculate_MIE()
    print("MIE: ", MIE)
    Ang, Gra = float("Nan"), float("Nan")
    NBC = landscape1.calculate_NBC()
    print("NBC: ", NBC)
    f = open(name,'a',newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow([found_file, FDC, FBD, PLO, Ske, Kur, CL, MIE, Ang, NBC, Gra, 0, 0])
    f.close()
    
    
if __name__ == "__main__":
    file_names=[]
    for home, dirs, files in os.walk('Data/'.format()):
        for filename in files:
            file_names.append(filename)
        file_names.sort()
    print(file_names)
    skiped_data = []  # Skipped data
    dir_datas = ['Data/{}'.format(file_name) for file_name in file_names]
    dir_landscapes = ['Landscape results/{}'.format(file_name[:-4]) for file_name in file_names]
    
    for dataset, lanscape_data in zip(dir_datas,dir_landscapes): # For each dataset
        search_string = 'None-Off'  # Replace with the string you want to search for
        if dataset in ['Data/Apache.csv','Data/redis.csv','Data/Hadoop.csv']:
            min_ = False
        else:
            min_ = True
        
        if search_string == 'None-Off':
            print('Dataset: ' + dataset)
            total_tasks = 1
            whole_data = genfromtxt(dataset, delimiter=',', skip_header=1)
            (N, n) = whole_data.shape
            n = n - 1
            non_zero_indexes = get_non_zero_indexes(whole_data, total_tasks)
            print('Total sample size: ', len(non_zero_indexes))
            N_features = n + 1 - total_tasks
            print('N_features: ', N_features)   
            
            unique_elements_per_column = [np.unique(whole_data[:, i]) for i in range(whole_data.shape[1]-1)]
            print("unique_elements_per_column_lens: ",len(unique_elements_per_column))
            
            # Get performance column data
            performance_column = whole_data[:, -1]  # Assume performance is the last column
            # Find minimum value in performance column
            if min_:
                min_performance = np.min(performance_column)
            else:
                min_performance = np.max(performance_column)
            # Filter rows where performance equals minimum value
            min_performance_rows = whole_data[performance_column == min_performance]
            # Get row data with minimum performance
            real_best = [row[:-1] for row in min_performance_rows] 

            lower_bound = [np.min(whole_data[:, i]) for i in range(whole_data.shape[1]-1)]
            upper_bound = [np.max(whole_data[:, i]) for i in range(whole_data.shape[1]-1)]
            k=0
            for i,j in zip(lower_bound,upper_bound):
                if i == j:
                    lower_bound[k]=lower_bound[k] - 1e-8
                k=k+1
            
            parts = dataset.split('/')
            data_name = parts[1]

            if not os.path.exists('./real_data/'):
                os.makedirs('./real_data/') # Create data storage folder

            name = './real_data/'+str(data_name)
            if not os.path.exists(name):
                f = open(name,'w',newline="")
                csv_writer = csv.writer(f)
                csv_writer.writerow(['Name','FDC','FBD','PLO','Ske','Kur','CL','MIE','Ang',"NBC",'Gra','MAPE','RD'])
                f.close()

                print(min_performance)
                run_main(whole_data, None, name, lower_bound, upper_bound, real_best, unique_elements_per_column, min_)
                
        print(skiped_data)