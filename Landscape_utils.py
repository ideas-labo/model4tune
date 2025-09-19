import random
import pandas as pd
import re
import os
import csv
import numpy as np
import math

from pflacco.misc_features import calculate_fitness_distance_correlation
from pflacco.classical_ela_features import *

def auto_correlation_at(x, N, lag):
    s = sum(x)
    mean = s / N
    start = max(0, -lag)
    limit = min(N, len(x))
    limit = min(limit, len(x) - lag)
    sum_corr = 0.0
    for i in range(start, limit):
        sum_corr += (x[i] - mean) * (x[i + lag] - mean)
    return sum_corr / (N - lag)

def calculate_sd(num_array):
    sum_num = sum(num_array)
    mean = sum_num / len(num_array)
    variance = sum((num - mean) ** 2 for num in num_array) / len(num_array)
    return math.sqrt(variance)

def hamming_dist(str1, str2):
    return sum(a != b for a, b in zip(str1, str2))

def find_files(directory, pattern):
    """
    Search for files containing a specific string in their names within the specified directory and its subdirectories.
    
    Args:
        directory: Directory to search
        pattern: Specific string contained in the filename
        
    Returns:
        List of matching files
    """
    matches = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if pattern in filename:
                matches.append(root+"/"+filename)
    return matches    

def csv_to_dict(filename):
    with open(filename, mode='r', encoding='utf-8') as csvfile:
        csv_data = csv.reader(csvfile)
        header = next(csv_data)  # Skip header
        my_dict = {int(rows[0]): rows[1] for rows in csv_data}
    return my_dict

def find_value_indices(lst, Min=True):
    if Min:
        min_value = min(lst)
    else:
        min_value = max(lst)
    # Find all indices with the minimum value
    min_indices = [index for index, value in enumerate(lst) if value == min_value]
    return min_indices

class landscape():
    def __init__(self, map, Min, df, last_column_values, lower_bound, upper_bound) -> None:
        self.map = map
        self.populations = list(map.keys())
        self.preds = list(map.values())
        self.Min = Min
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.df = df 
        self.last_column_values = last_column_values
        self.best = find_value_indices(self.preds, Min=self.Min)
        self.best_populations = [self.populations[i] for i in self.best]
        self.ela_distr = calculate_ela_distribution(df, last_column_values)
        self.ic = None

    def _get_auto_correlation(self):
        list_keys = self.populations[:]
        base = self.best_populations
        def _min_distance(x, base):
            dis = []
            for single_base in base:
                x_best = single_base
                if hamming_dist(x_best, x) != 0:
                    dis.append(hamming_dist(x_best, x))
            if dis:
                return min(dis)
            else:
                return 1e8
        list_keys.sort(key=lambda x: _min_distance(x, base))
        total = 0
        size = 50
        for _ in range(size):
            sub_list = base[:]
            first = random.choice(base)
            p = random.randint(0, len(list_keys) - 1)
            for _ in range(50):
                # Random walk excluding itself
                k=1
                temp = []
                while len(temp) == 0 or len(temp) == 1:  # Prevent no neighbors
                    temp = [list_keys[i] for i in range(len(list_keys)) if hamming_dist(list_keys[i], first) <= k]
                    k += 1
                p = random.randint(0, len(temp) - 1)
                sub_list.append(temp[p])
                first = temp[p]
            data = [self.map[key] for key in sub_list]
            r = auto_correlation_at(data, len(data), 1)
            std = calculate_sd(data)
            if std == 0:
                size = size - 1
                continue
            total += r / (std * std)
        return total / size

    def calculate_correlation_length(self):
        d = self._get_auto_correlation()
        if d == 0 or abs(d) == 1:
            return "nan"
        return (1 / math.log(abs(d))) * -1.0

    def calculate_best_distance(self, real_bests):
        base = self.best_populations
        dis = []
        for real_best in real_bests:
            for single_list in base:
                dis.append(hamming_dist(single_list, real_best))
        return min(dis)

    def calculate_Proportion_of_local_optimal(self, unique_elements_per_column):
        def _is_better(x_value, y_value, minimization):
            if minimization:
                return x_value <= y_value
            else:
                return x_value >= y_value   
            
        def _find_neighbourhood(configuration,unique_elements_per_column):
            res = []
            configuration = list(configuration)
            for i in range(len(unique_elements_per_column)):
                if len(unique_elements_per_column[i])>5:
                    tmp_independent_set = random.sample(list(unique_elements_per_column[i]),5)
                else:
                    tmp_independent_set = list(unique_elements_per_column[i])
                for j in tmp_independent_set:
                    tmp = configuration[:]
                    tmp[i] = j
                    res.append(tuple(tmp))
            random.shuffle(res)
            return res
        
        tmp = 0
        population_count = len(self.populations)

        for i in self.populations:
            flag = 1
            for j in _find_neighbourhood(i, unique_elements_per_column):
                if j in self.map:
                    if not _is_better(self.map[i], self.map[j], self.Min):
                        flag = 0
                        break  # Found local optimum, break loop
            if flag:
                tmp += 1  # Count
        return tmp / population_count  # Return proportion of local optima

    def calculate_FDC(self):
        fdc = calculate_fitness_distance_correlation(self.df, self.last_column_values, minimize=self.Min)
        return fdc['fitness_distance.fd_correlation']
    
    def calculate_skewness(self):
        return self.ela_distr['ela_distr.skewness']
    
    def calculate_kurtosis(self):
        return self.ela_distr['ela_distr.kurtosis']
    
    def calculate_MIE(self):
        ic = calculate_information_content(self.df, self.last_column_values, seed = 100)
        self.ic = ic 
        return ic['ic.h_max']

    def calculate_eps_s(self):
        if self.ic:
            return self.ic['ic.eps_s']
        else:
            return calculate_information_content(self.df, self.last_column_values, seed = 100)['ic.eps_s']
    
    def calculate_NBC(self):
        nbc = calculate_nbc(self.df, self.last_column_values)
        return nbc['nbc.nn_nb.mean_ratio']
    
    def calculate_Gradient_Homogeneity(self):
        cm_grad = calculate_cm_grad(self.df, self.last_column_values, lower_bound=self.lower_bound,
                          upper_bound=self.upper_bound, blocks=2, force=True, minimize=self.Min)
        return cm_grad['cm_grad.mean']
    
    def calculate_Angle(self):
        cm_angle = calculate_cm_angle(self.df, self.last_column_values, lower_bound=self.lower_bound,
                          upper_bound=self.upper_bound, blocks=2, force=True, minimize=self.Min)
        return cm_angle["cm_angle.angle_mean"]
    
def RD(y_test, y_pred):
    y_test_return = []
    y_pred_retuen = []
    y_test_sort = sorted(y_test)
    y_pred_sort = sorted(y_pred)
    for i,val in enumerate(y_test):
        y_test_return.append(y_test_sort.index(val)+1)
    for i,val in enumerate(y_pred):
        y_pred_retuen.append(y_pred_sort.index(val)+1)   
    dist = np.sum(np.abs(np.array(y_test_return) - np.array(y_pred_retuen)))/len(y_test_return)
    
    return dist

def MAPE(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    errors = np.abs((actual - predicted) / actual)
    mape = np.mean(errors) * 100
    return mape

def run_main(whole_data, found_file, name, lower_bound, upper_bound, real_best, unique_elements_per_column, min_):

    seed = int(re.findall(r'Run(\d+)', found_file)[0])
    random.seed(seed)
    np.random.seed(seed)
    data_dict = csv_to_dict(found_file)
    print("Now data: ", found_file)
    # Since training set was selected for model training, need to reconstruct original data
    # Index numbers are original index numbers, index the sequence number to the original dataframe
    map = {tuple(whole_data[n][:-1]): float(data_dict[n]) for n in data_dict.keys()}

    # Calculate accuracy metrics MAPE and RD
    map_real = {tuple(whole_data[n][:-1]): float(whole_data[n][-1]) for n in data_dict.keys()}
    RD_ = RD(list(map_real.values()), list(map.values()))
    MAPE_ = MAPE(list(map_real.values()), list(map.values()))
    print("RD:", RD_)
    print("MAPE:", MAPE_)

    # Start converting to format for API use [df: configs ===== last_column_values: values]
    last_column_values = pd.Series(list(map.values()))
    numpy_array = np.array(list(map.keys()))
    # Convert numpy array to pandas DataFrame
    df = pd.DataFrame(numpy_array)
    # Set the first column as index column
    df.index = pd.RangeIndex(start=0, stop=len(df), step=1)

    landscape1 = landscape(map=map, Min=min_, df=df, last_column_values=last_column_values, lower_bound=lower_bound, upper_bound=upper_bound)

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
    NBC = landscape1.calculate_NBC()
    print("NBC: ", NBC)
    Ang = np.nan
    print("Angle: ", Ang)
    Gra = np.nan
    print("Gradient Homogeneity: ", Gra)
    f = open(name,'a',newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow([found_file, FDC, FBD, PLO, Ske, Kur, CL, MIE, Ang, NBC, Gra, MAPE_, RD_])
    f.close()