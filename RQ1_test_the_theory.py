import pandas as pd
import numpy as np
from itertools import product
from scipy import stats

# Read data
df = pd.read_csv('./processed_data/performance_with_metrics.csv')

# Define two feature groups
global_metrics = ['FDC', 'FBD', 'Ske', 'Kur']  # Global metrics group
direct_metrics = ['PLO', 'CL', 'MIE', 'NBC']  # Direct metrics group

# Define grouping criteria: system and OR/Ac/SH features
grouping_cols = ['System', 'OR_0', 'OR_1', 'OR_2', 'OR_3', 'Ac_0', 'Ac_1', 'Ac_2', 'Ac_3', 'SH']

# Define system order (standard case format)
system_order = [
    'Apache', '7z', 'DConvert', 'DeepArch', 'ExaStencils',
    'Hadoop', 'MariaDB', 'MongoDB', 'PostgreSQL', 'Redis', 'SPARK',
    'Storm', 'HSMGP', 'XGBoost', 'HIPAcc', 'SQL', 'JavaGC', 'Polly'
]

def dominates(row1, row2, features):
    """
    Check if row1 dominates row2
    Domination condition: all feature values are less than or equal to the counterpart, 
    and at least one feature value is strictly less than the counterpart
    """
    better_count = 0
    worse_count = 0
    
    for feature in features:
        if pd.isna(row1[feature]) or pd.isna(row2[feature]):
            continue
        if row1[feature] < row2[feature]:
            better_count += 1
        elif row1[feature] > row2[feature]:
            worse_count += 1
    
    # If any feature is worse, then it doesn't dominate
    if worse_count > 0:
        return False
    # If at least one feature is better, then it dominates
    return better_count > 0

def calculate_domination_analysis_subset(group_df, feature_subset):
    """Calculate domination relationship analysis based on feature subset in the given grouped dataframe"""
    n = len(group_df)
    dominated = [False] * n
    
    # Check each pair of data points
    for i in range(n):
        for j in range(n):
            if i != j:
                if dominates(group_df.iloc[i], group_df.iloc[j], feature_subset):
                    dominated[j] = True
    
    # Classify as dominating and non-dominating
    dominating_indices = []
    dominated_indices = []
    non_dominated_indices = []
    
    for i in range(n):
        if dominated[i]:
            dominated_indices.append(i)
        else:
            # Check if it dominates at least one other point
            is_dominating = False
            for j in range(n):
                if i != j and dominates(group_df.iloc[i], group_df.iloc[j], feature_subset):
                    is_dominating = True
                    break
            if is_dominating:
                dominating_indices.append(i)
            else:
                non_dominated_indices.append(i)
    
    return dominating_indices, dominated_indices, non_dominated_indices

def pad_lists_to_same_length(list1, list2):
    """Pad two lists to the same length for statistical testing"""
    max_len = max(len(list1), len(list2))
    
    # Use repeated sampling to pad the shorter list
    if len(list1) < max_len:
        if len(list1) > 0:
            list1_padded = np.random.choice(list1, max_len, replace=True)
        else:
            list1_padded = []
    else:
        list1_padded = list1
    
    if len(list2) < max_len:
        if len(list2) > 0:
            list2_padded = np.random.choice(list2, max_len, replace=True)
        else:
            list2_padded = []
    else:
        list2_padded = list2
    
    return list1_padded, list2_padded

# Generate feature combinations: each combination contains one global metric and one direct metric
feature_combinations = list(product(global_metrics, direct_metrics))

print(f"Global metrics group: {global_metrics}")
print(f"Direct metrics group: {direct_metrics}")
print(f"Number of generated feature combinations: {len(feature_combinations)}")

# Group by system and OR/Ac/SH features
print(f"\nGrouping criteria: {grouping_cols}")

# Create group identifiers
df['group_id'] = df[grouping_cols].apply(lambda x: '_'.join(map(str, x)), axis=1)
groups = df.groupby('group_id')

print(f"Total number of groups: {len(groups)}")

# Perform domination relationship analysis by group
all_results = []
processed_groups = 0
valid_groups = 0

# Store all dominating and non-dominated point performance for each system
system_dominating_perf = {}
system_non_dominated_perf = {}

def calculate_domination_pairs(group_df, feature_subset):
    """
    Calculate pairwise domination relationships in the given grouped dataframe based on feature subset
    Returns list of domination pairs [(dominating point index, dominated point index)]
    """
    n = len(group_df)
    domination_pairs = []
    
    # Check each pair of data points
    for i in range(n):
        for j in range(n):
            if i != j:
                # If i dominates j, add it as a domination pair
                if dominates(group_df.iloc[i], group_df.iloc[j], feature_subset):
                    domination_pairs.append((i, j))
    
    return domination_pairs

# Create system name mapping dictionary to map system names in data to standard case
system_name_mapping = {}
for system in system_order:
    system_name_mapping[system.upper()] = system
    system_name_mapping[system.lower()] = system
    system_name_mapping[system] = system

# Modify processing logic in main loop
for group_id, group_df in groups:
    processed_groups += 1
    
    # Reset index
    group_df = group_df.reset_index(drop=True)
    
    # Remove rows containing NaN values
    feature_cols_all = global_metrics + direct_metrics
    group_df_clean = group_df.dropna(subset=feature_cols_all + ['perf'])
    
    if len(group_df_clean) < 2:
        continue
    
    valid_groups += 1
    original_system = group_df_clean['System'].iloc[0]
    
    # Standardize system name (ensure case consistency with system_order)
    system = system_name_mapping.get(original_system, original_system)
    
    # Initialize system performance storage
    if system not in system_dominating_perf:
        system_dominating_perf[system] = []
        system_non_dominated_perf[system] = []
    
    group_results = []
    
    for feature_combo in feature_combinations:
        try:
            # Use new function to get pairwise domination pairs
            domination_pairs = calculate_domination_pairs(group_df_clean, feature_combo)
            
            if not domination_pairs:
                continue
                
            # Calculate performance statistics
            all_perf = group_df_clean['perf'].values
            
            # Collect performance data of dominating and dominated points (one-to-one)
            dominating_perf = []
            dominated_perf = []
            
            for dom_idx, dominated_idx in domination_pairs:
                dominating_perf.append(all_perf[dom_idx])
                dominated_perf.append(all_perf[dominated_idx])
            
            # Collect performance data of domination pairs (ensure one-to-one matching)
            system_dominating_perf[system].extend(dominating_perf)
            system_non_dominated_perf[system].extend(dominated_perf)
            
            # Calculate statistical information
            result = {
                'group_id': group_id,
                'system': system,  # Use standardized system name
                'feature_combo': feature_combo,
                'total_points': len(group_df_clean),
                'domination_pairs_count': len(domination_pairs),
                'dominating_perf_mean': np.mean(dominating_perf) if dominating_perf else None,
                'dominated_perf_mean': np.mean(dominated_perf) if dominated_perf else None,
                'all_perf_mean': np.mean(all_perf),
            }
            
            # Calculate performance difference (dominating point performance - dominated point performance)
            if result['dominating_perf_mean'] is not None and result['dominated_perf_mean'] is not None:
                result['perf_diff_dom_vs_dominated'] = result['dominating_perf_mean'] - result['dominated_perf_mean']
            else:
                result['perf_diff_dom_vs_dominated'] = None
            
            # Add additional statistical information
            result['unique_dominating_count'] = len(set(dom_idx for dom_idx, _ in domination_pairs))
            result['unique_dominated_count'] = len(set(dominated_idx for _, dominated_idx in domination_pairs))
            
            group_results.append(result)
            
        except Exception as e:
            continue
    
    all_results.extend(group_results)
    
print(f"\nProcessing completed:")
print(f"Total groups: {processed_groups}")
print(f"Valid groups (>=2 data points): {valid_groups}")

# Convert to DataFrame for analysis
results_df = pd.DataFrame(all_results)

print("\n" + "="*80)
print("System Performance Difference Analysis Results (LaTeX Table)")
print("="*80)

if not results_df.empty:
    # Filter valid results
    valid_results_df = results_df[results_df['perf_diff_dom_vs_dominated'].notna()].copy()
    
    print(f"Total result count: {len(results_df)}")
    print(f"Valid result count (with domination relationships): {len(valid_results_df)}")
    
    if len(valid_results_df) > 0:
        # Calculate statistical information for each system
        latex_table_data = []
        
        for system in system_order:
            # Check if system exists in data
            if system not in valid_results_df['system'].unique():
                # If system doesn't exist in data, add empty data
                latex_table_data.append({
                    'System': system,
                    'Mean_Diff': np.nan,
                    'Median_Diff': np.nan,
                    'Dominating_Better_Ratio': np.nan,
                    'Non_Dominated_Better_Ratio': np.nan,
                    'P_Value': np.nan,
                    'Sample_Size': 0
                })
                continue
                
            system_data = valid_results_df[valid_results_df['system'] == system]['perf_diff_dom_vs_dominated']
            
            # Basic statistics
            mean_diff = system_data.mean()
            median_diff = system_data.median()
            
            # Calculate ratios where dominating is better and non-dominated is better
            # Since performance is better when smaller:
            # Dominating better: perf_diff < 0 (dominating point performs better)
            # Non-dominated better: perf_diff > 0 (non-dominated point performs better)
            dominating_better_count = len(system_data[system_data < 0])  # Dominating point performs better
            non_dominated_better_count = len(system_data[system_data > 0])  # Non-dominated point performs better
            total_count = len(system_data)
            
            dominating_better_ratio = dominating_better_count / total_count * 100 if total_count > 0 else 0
            non_dominated_better_ratio = non_dominated_better_count / total_count * 100 if total_count > 0 else 0
            
            # Special handling for Redis, Hadoop, Apache systems - reverse the results
            if system.lower() in ['redis', 'hadoop', 'apache']:
                median_diff = -median_diff
                mean_diff = -mean_diff
                # Swap the ratios
                dominating_better_ratio, non_dominated_better_ratio = non_dominated_better_ratio, dominating_better_ratio
            
            # Perform statistical test (U-test)
            dominating_perf_sys = system_dominating_perf.get(system, [])
            non_dominated_perf_sys = system_non_dominated_perf.get(system, [])
            
            p_value = None
            
            if len(dominating_perf_sys) > 0 and len(non_dominated_perf_sys) > 0:
                # Pad to same length
                dom_padded, non_dom_padded = pad_lists_to_same_length(dominating_perf_sys, non_dominated_perf_sys)
                
                if len(dom_padded) > 0 and len(non_dom_padded) > 0:
                    try:
                        test_stat, p_value = stats.mannwhitneyu(dom_padded, non_dom_padded, alternative='two-sided')
                        print(f"Statistical test successful {system}: U={test_stat}, p={p_value}")
                    except Exception as e:
                        print(f"Statistical test failed {system}: {e}")
                        p_value = None
            
            # Determine marker symbol based on p-value
            marker = ""
            if p_value is not None:
                if p_value < 0.001:
                    marker = "$^\\dagger$"
                elif p_value < 0.05:
                    marker = "$^\\star$"
                else:
                    marker = "$^{\\,\\,\\,}$"
            
            latex_table_data.append({
                'System': system,
                'Mean_Diff': mean_diff,
                'Median_Diff': median_diff,
                'Dominating_Better_Ratio': dominating_better_ratio,
                'Non_Dominated_Better_Ratio': non_dominated_better_ratio,
                'P_Value': p_value,
                'P_Marker': marker,
                'Sample_Size': total_count
            })
        
        # Display detailed statistical information
        print("\n" + "="*80)
        print("Detailed Statistical Information")
        print("="*80)
        
        detail_df = pd.DataFrame(latex_table_data)
        # Only display systems with valid data
        valid_detail_df = detail_df[~detail_df['Median_Diff'].isna()]
        
        print("Performance by system order:")
        print(valid_detail_df[['System', 'Median_Diff', 'Non_Dominated_Better_Ratio', 
                           'Dominating_Better_Ratio', 'P_Value', 'Sample_Size']].to_string(index=False))
        
        # Overall statistics
        print(f"\nOverall statistics:")
        all_mean_diff = valid_results_df['perf_diff_dom_vs_dominated'].mean()
        all_median_diff = valid_results_df['perf_diff_dom_vs_dominated'].median()
        
        all_dominating_better = len(valid_results_df[valid_results_df['perf_diff_dom_vs_dominated'] < 0])
        all_non_dominated_better = len(valid_results_df[valid_results_df['perf_diff_dom_vs_dominated'] > 0])
        all_total = len(valid_results_df)
        
        print(f"Overall average performance difference: {all_mean_diff:.4f}")
        print(f"Overall median performance difference: {all_median_diff:.4f}")
        print(f"Overall dominating better ratio: {all_dominating_better/all_total*100:.1f}%")
        print(f"Overall non-dominated better ratio: {all_non_dominated_better/all_total*100:.1f}%")
        
        
    else:
        print("No valid domination relationships found")

else:
    print("No results generated")
