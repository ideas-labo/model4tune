# -*- coding: utf-8 -*-
import re
import os
import numpy as np
import multiprocessing as mp
import multiprocessing as mp

from collections import Counter
from numpy import genfromtxt
from utils.general import get_non_zero_indexes
from pflacco.classical_ela_features import *
from Landscape_utils import *

if __name__ == "__main__":
    file_names=[]
    for home, dirs, files in os.walk('Data/'.format()):
        for filename in files:
            file_names.append(filename)
        file_names.sort()
    
    skiped_data = []  # Skipped data list
    dir_datas = ['Data/{}'.format(file_name) for file_name in file_names]
    dir_landscapes = ['results/{}'.format(file_name[:-4]) for file_name in file_names]
    # mp.freeze_support()
    # pool = mp.Pool(processes=10)
    for dataset, landscape_data in zip(dir_datas[::-1],dir_landscapes[::-1]): # For each dataset
        if dataset in ['Data/Apache.csv', 'Data/redis.csv', 'Data/Hadoop.csv']:
            min_ = False
        else:
            min_ = True

        whole_data = genfromtxt(dataset, delimiter=',', skip_header=1)
        (N, n) = whole_data.shape
        n = n - 1
        N_features = n
        search_strings = ["None"]
        search_strings.extend(["F"+"%d" %i+"-Off" for i in range(N_features)])
        # search_strings = [""]
        print("==> Search stings: ", search_strings)
        for search_string in search_strings:
            if search_string == "None":
                off_feature = search_string
                pass
            else:
                off_feature = int(re.findall(r'\d+', search_string)[0])
                temp_delete = []
                counter = Counter(whole_data[:, off_feature])
                for temp_i, temp_sample in enumerate(whole_data):
                    if counter.most_common(1)[0][1] > 5000:
                        if temp_sample[off_feature] != np.min(whole_data[:, off_feature]):
                            temp_delete.append(int(temp_i))
                    else:
                        if temp_sample[off_feature] != counter.most_common(1)[0][0]:
                            temp_delete.append(int(temp_i))

                whole_data = np.delete(whole_data, temp_delete, axis=0)
            (N, n) = whole_data.shape
            n = n - 1

            print('==> Dataset: ' + dataset)
            total_tasks = 1

            non_zero_indexes = get_non_zero_indexes(whole_data, total_tasks)
            print('==> Total sample size: ', len(non_zero_indexes))
            N_features = n + 1 - total_tasks
            # if N_features >= 20:  # memory out 
            #     skiped_data.append(dataset)
            #     continue
            print('==> N_features: ', N_features)
            # print(whole_data[0])
            unique_elements_per_column = [np.unique(whole_data[:, i]) for i in range(whole_data.shape[1]-1)]
            print("==> unique_elements_per_column_lens: ",len(unique_elements_per_column))
            
            # Get performance column data
            performance_column = whole_data[:, -1]  
            # Find minimum value in performance column
            if min_:
                min_performance = np.min(performance_column)
            else:
                min_performance = np.max(performance_column)
            # Filter rows where performance equals minimum value
            min_performance_rows = whole_data[performance_column == min_performance]
            # Print row data with minimum performance
            real_best = [row[:-1] for row in min_performance_rows] 

            lower_bound = [np.min(whole_data[:, i]) for i in range(whole_data.shape[1]-1)]
            upper_bound = [np.max(whole_data[:, i]) for i in range(whole_data.shape[1]-1)]
            k=0
            for i,j in zip(lower_bound,upper_bound):
                if i == j:
                    lower_bound[k]=lower_bound[k] - 1e-8
                k=k+1

            for root, dirs, files in os.walk(str(landscape_data)+'/'.format()):
                print(f"---|Now root: {root}")
                # Open each subdirectory
                for dir in dirs[:]:
                    # Get full path of subdirectory
                    sub_dir = os.path.join(root, dir)
                    print(f"---|---||Child root: {sub_dir}")
                    
                    found_files = find_files(sub_dir, search_string)
                    
                    parts = sub_dir.split('/')
                    data_name = parts[1]
                    model_name = parts[2]
                    if not os.path.exists('./results_individual/'+str(data_name)):
                        os.makedirs('./results_individual/'+str(data_name)) # Create data storage folder
                    if not os.path.exists('./results_individual/'+str(data_name)+'/'+str(off_feature)):
                        os.makedirs('./results_individual/'+str(data_name)+'/'+str(off_feature)) # Create data storage folder
                    if model_name != "Perf-AL":  # Perf-AL is skipped
                        name = './results_individual/'+str(data_name)+'/'+str(off_feature)+'/'+str(model_name)+'.csv'
                        if not os.path.exists(name):
  
                            pool = mp.Pool()
                            for found_file in found_files:  # Contains thirty random seeds
                                run_main(whole_data, found_file, name, lower_bound, upper_bound, real_best, unique_elements_per_column, min_)
                                # pool.apply_async(run_main,(whole_data, found_file, name, lower_bound, upper_bound, real_best, unique_elements_per_column, min_))
                            # pool.close()
                            # pool.join()
            # print(skiped_data)
                    

    # pool.close()