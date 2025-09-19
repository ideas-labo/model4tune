import pandas as pd
import numpy as np
import os
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

data_set_name = './landscape_data/difference'
directory_path = f"{data_set_name}"
files_origin = [f for f in os.listdir(directory_path)]
files = [os.path.join(directory_path, file) for file in files_origin]

data_folder_path = "./landscape_data/difference"

def perform_knob_clustering(data, col_index, n_clusters=2):
    """
    Perform clustering on entire knob vectors for a specific column
    
    Parameters:
    - data: DataFrame containing all data
    - col_index: Column to use for clustering
    - n_clusters: Number of clusters to form (default is 2)
    
    Returns:
    - cluster_labels: Array of cluster labels for each knob
    - cluster_centers: Centroids of the clusters
    """
    # Group data by first column (knob)
    grouped = data.groupby(0)
    
    # Prepare feature vectors for each knob
    knob_vectors = []
    knob_names = []
    
    for knob, group in grouped:
        # Remove rows with NaN or inf values
        group_cleaned = group.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(group_cleaned) > 0:
            # Use the specified column as feature vector
            feature_vector = group_cleaned[col_index].values
            knob_vectors.append(np.mean(feature_vector))  # Use mean of column values
            knob_names.append(knob)
    
    # Convert to numpy array
    X = np.array(knob_vectors).reshape(-1, 1)
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    return cluster_labels, kmeans.cluster_centers_, knob_names



# Process each file
for index, file in enumerate(files):
    # Read data
    data = pd.read_csv(file, header=None)
    columns_to_extract = list(range(0, 12))
    data = data[columns_to_extract]
    
    # Read header data
    data_header_file = os.path.join(data_folder_path, files_origin[index])
    header_data = pd.read_csv(data_header_file).columns.tolist()
    
    # Prepare y_values_str
    y_values_str = []
    for y in range(len(header_data)):
        try:
            y_values_str.append(header_data[y])
        except:
            y_values_str.append(f'Unknown_{y}')
    
    # Perform clustering for each column from 2 to 11
    print(f"\nClustering Results for {files_origin[index]}:")
    y_values_str_land = ['FDC','FBD','PLO','Ske','Kur','CL','h_max','NBC','MAPE','RD']
    for col_index in range(2, 11):
        print(f"\nClustering for Column {col_index} ({y_values_str_land[col_index-2]})")
        # Perform clustering
        cluster_labels, cluster_centers, knob_names = perform_knob_clustering(data, col_index)
        names = [y_values_str[int(i)] if i != 'None1' and int(i) < len(y_values_str) else i for i in [knob_names[j] for j in range(len(knob_names))]]         
        cluster_labels1 = [y_values_str_land[col_index-2]]+[0 if i == cluster_labels[-1] else 1 for i in cluster_labels]
        print(cluster_labels1)
        # Print cluster distribution
        for cluster in range(len(np.unique(cluster_labels))):
            cluster_knobs = [y_values_str[int(i)] if i != 'None1' and int(i) < len(y_values_str) else i for i in [knob_names[j] for j in range(len(knob_names)) if cluster_labels[j] == cluster]]

    print("-" * 50)
