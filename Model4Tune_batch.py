# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging
from scipy.stats import mannwhitneyu
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration class for experiment parameters"""
    max_rank: int = 10
    n_iterations: int = 5  
    base_learning_rate: float = 0.05  
    learning_rate_step: float = 0.01  
    data_path: str = './processed_data/M4T_B.csv'
    
    # Model parameters
    model_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.model_params is None:
            self.model_params = {
                'learning_rate': self.base_learning_rate,  
                'max_depth': 6,
                'min_child_samples': 30,
                'n_estimators': 100,
                'num_leaves': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 0.5,
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'verbose': -1,
                'seed': 42  
            }


class FeatureManager:
    """Manages feature groups and combinations for batch data"""
    
    def __init__(self):
        self.base_features = ['Online'] + \
                            [f'Domain_{i}' for i in range(4)] + \
                            [f'Tech_{i}' for i in range(4)] + \
                            ['MAPE', 'RD']
        
        self.global_metrics = ['FDC', 'FBD', 'Ske', 'Kur']
        self.direct_metrics = ['PLO', 'CL', 'MIE', 'NBC']
    
    def generate_feature_combinations(self) -> List[Tuple[List[str], str]]:
        """Generate all possible feature combinations"""
        combinations = []
        for global_metric in self.global_metrics:
            for direct_metric in self.direct_metrics:
                features = self.base_features + [global_metric, direct_metric]
                combo_name = f"{global_metric}+{direct_metric}"
                combinations.append((features, combo_name))
        return combinations


class RankingUtilities:
    """Utility functions for ranking and evaluation"""
    
    @staticmethod
    def convert_to_rank_per_system(y, systems, max_rank=10):
        y_rank = np.zeros_like(y, dtype=int)
        unique_systems = np.unique(systems)
        reverse_systems = ['Apache', 'redis', 'Hadoop']
        
        for system in unique_systems:
            mask = systems == system
            system_y = y[mask]
            
            if len(system_y) > 1:
                ascending = system not in reverse_systems
                if ascending:
                    ranks = np.argsort(np.argsort(-system_y))
                else:
                    ranks = np.argsort(np.argsort(system_y))
                
                if len(ranks) > 1:
                    normalized_ranks = (ranks * max_rank // (len(ranks) - 1)).astype(int)
                    normalized_ranks = np.clip(normalized_ranks, 0, max_rank)
                else:
                    normalized_ranks = np.array([0])
                y_rank[mask] = normalized_ranks
            else:
                y_rank[mask] = 0
        return y_rank
    
    @staticmethod
    def calculate_ndcg_per_system(y_true: np.ndarray, y_pred: np.ndarray, 
                                systems: np.ndarray, k: Optional[int] = None) -> float:
        """Calculate NDCG@k for each system"""
        ndcg_scores = []
        unique_systems = np.unique(systems)
        
        for system in unique_systems:
            mask = systems == system
            if np.sum(mask) > 1:
                system_true = y_true[mask]
                system_pred = y_pred[mask]
                
                try:
                    ndcg = ndcg_score([system_true], [system_pred], k=k)
                    ndcg_scores.append(ndcg)
                except:
                    ndcg_scores.append(0.0)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    @staticmethod
    def calculate_map_per_system(y_true: np.ndarray, y_pred: np.ndarray, 
                               systems: np.ndarray, k: Optional[int] = None) -> float:
        """Calculate AP@k (Mean Average Precision) for each system"""
        map_scores = []
        unique_systems = np.unique(systems)
        
        for system in unique_systems:
            mask = systems == system
            if np.sum(mask) > 1:
                system_true = y_true[mask]
                system_pred = y_pred[mask]
                
                pred_indices = np.argsort(-system_pred)
                
                if k is not None:
                    pred_indices = pred_indices[:k]
                else:
                    k = len(pred_indices)
                
                relevance_threshold = np.median(system_true)
                binary_true = (system_true > relevance_threshold).astype(int)
                
                precision_sum = 0.0
                num_relevant = 0
                
                for i, idx in enumerate(pred_indices):
                    if binary_true[idx] == 1:
                        num_relevant += 1
                        precision_sum += num_relevant / (i + 1)
                
                ap = precision_sum / max(1, k) if k > 0 else 0.0
                map_scores.append(ap)
        
        return np.mean(map_scores) if map_scores else 0.0


class ModelEvaluator:
    """Handles model training and evaluation"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.ranking_utils = RankingUtilities()
    
    def evaluate_single_combination(self, X: np.ndarray, y_rank: np.ndarray, 
                                  systems: np.ndarray, combo_name: str) -> List[Dict[str, Any]]:
        """Evaluate a single feature combination using LOSO cross-validation"""
        results = []
        unique_systems = np.unique(systems)
        
        for test_system in unique_systems:
            logger.info(f"Testing system: {test_system}")
            
            train_mask = systems != test_system
            test_mask = systems == test_system
            
            X_train, y_train, systems_train = X[train_mask], y_rank[train_mask], systems[train_mask]
            X_test, y_test, systems_test = X[test_mask], y_rank[test_mask], systems[test_mask]
            
            if len(X_test) < 2:
                continue
            
            train_group_sizes = [np.sum(systems_train == system) for system in np.unique(systems_train)]
            
            system_results = self._evaluate_multiple_learning_rates(
                X_train, y_train, train_group_sizes, X_test, y_test, 
                systems_test, combo_name, test_system
            )
            results.extend(system_results)
        
        return results
    
    def _evaluate_multiple_learning_rates(self, X_train: np.ndarray, y_train: np.ndarray, 
                                        train_group_sizes: List[int], X_test: np.ndarray, 
                                        y_test: np.ndarray, systems_test: np.ndarray, 
                                        combo_name: str, test_system: str) -> List[Dict[str, Any]]:
        """Evaluate model with multiple learning rates instead of seeds"""
        results = []
        
        for iteration in range(self.config.n_iterations):
            current_lr = self.config.base_learning_rate + (iteration * self.config.learning_rate_step)
            
            try:
                model_params = self.config.model_params.copy()
                model_params['learning_rate'] = current_lr
                
                model = lgb.LGBMRanker(**model_params)
                model.fit(X_train, y_train, group=train_group_sizes)
                
                test_pred = model.predict(X_test)
                metrics = self._calculate_all_metrics(y_test, test_pred, systems_test)
                
                result = {
                    'feature_combo': combo_name,
                    'test_system': test_system,
                    'iteration': iteration,
                    'learning_rate': current_lr,
                    **metrics
                }
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error in iteration {iteration} (lr={current_lr:.3f}) for system {test_system}: {e}")
                continue
        
        return results
    
    def _calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             systems: np.ndarray) -> Dict[str, float]:
        """Calculate all evaluation metrics"""
        metrics = {}
        k_values = [1, 10, 20, None]
        
        for k in k_values:
            k_suffix = str(k) if k is not None else 'all'
            metrics[f'ndcg_{k_suffix}'] = self.ranking_utils.calculate_ndcg_per_system(y_true, y_pred, systems, k)
            metrics[f'map_{k_suffix}'] = self.ranking_utils.calculate_map_per_system(y_true, y_pred, systems, k)
        
        return metrics
    
    def evaluate_random_baseline(self, y_rank: np.ndarray, systems: np.ndarray) -> List[Dict[str, Any]]:
        """Evaluate random baseline performance with different random seeds"""
        logger.info("Calculating random baseline performance...")
        results = []
        unique_systems = np.unique(systems)
        
        for test_system in unique_systems:
            test_mask = systems == test_system
            y_test = y_rank[test_mask]
            systems_test = systems[test_mask]
            
            if len(y_test) < 2:
                continue
            
            for iteration in range(self.config.n_iterations):
                random_seed = 42 + iteration
                np.random.seed(random_seed)
                test_pred_random = np.random.random(len(y_test))
                metrics = self._calculate_all_metrics(y_test, test_pred_random, systems_test)
                
                result = {
                    'feature_combo': 'random',
                    'test_system': test_system,
                    'iteration': iteration,
                    'learning_rate': None, 
                    **metrics
                }
                results.append(result)
        
        return results


class ResultsAnalyzer:
    """Handles results analysis and visualization"""
    
    def __init__(self, feature_manager: FeatureManager):
        self.feature_manager = feature_manager
        self.system_order = [
            'Apache', '7z', 'DConvert', 'DeepArch', 'ExaStencils',
            'Hadoop', 'MariaDB', 'MongoDB', 'PostgreSQL', 'Redis', 'Spark',
            'Storm', 'HSMGP', 'XGBoost', 'HIPAcc', 'SQL', 'JavaGC', 'Polly'
        ]
        self.system_latex_names = {name.lower(): f"\\textsc{{{name}}}" for name in self.system_order}
    
    def _initialize_system_info(self, results_df: pd.DataFrame):
        """Initialize system information from results"""
        available_systems_lower = set(system.lower() for system in results_df['test_system'].unique())
    
        filtered_systems = []
        for system in self.system_order:
            if system.lower() in available_systems_lower:
                filtered_systems.append(system)
        
        self.system_order = filtered_systems
        
        self.system_name_mapping = {}
        actual_systems = results_df['test_system'].unique()
        
        for actual_system in actual_systems:
            for standard_system in self.system_order:
                if actual_system.lower() == standard_system.lower():
                    self.system_name_mapping[actual_system] = standard_system
                    break
        
        self.system_latex_names = {name.lower(): f"\\textsc{{{name}}}" for name in self.system_order}
    
    def analyze_results(self, results_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Analyze experiment results"""
        # Initialize system info
        self._initialize_system_info(results_df)
        
        metrics = ['ndcg_1', 'ndcg_10', 'ndcg_20', 'ndcg_all', 'map_1', 'map_10', 'map_20', 'map_all']
        summary = results_df.groupby('feature_combo')[metrics].mean().round(4)
        
        # Calculate improvement relative to random
        random_performance = summary.loc['random']
        improvement = summary.copy()
        
        for feature_combo in summary.index:
            if feature_combo != 'random':
                for metric in metrics:
                    improvement.loc[feature_combo, metric] = (
                        summary.loc[feature_combo, metric] / random_performance[metric] - 1
                    ) * 100
        
        return summary, improvement
    
    def generate_feature_combination_tables(self, results_df: pd.DataFrame):
        """Generate individual LaTeX tables for each feature combination"""
        results_df = results_df.copy()
        results_df['type'] = results_df['feature_combo'].apply(lambda x: 'random' if x == 'random' else 'model')
        results_df['test_system_lower'] = results_df['test_system'].str.lower()
        
        metrics = ['ndcg_1', 'ndcg_10', 'ndcg_20', 'ndcg_all', 'map_1', 'map_10', 'map_20', 'map_all']
        
        import os
        os.makedirs('feature_combo_tables_batch', exist_ok=True)

        feature_combos = results_df[results_df['type'] == 'model']['feature_combo'].unique()
    
        for combo in feature_combos:
            combo_data = results_df[results_df['feature_combo'] == combo]
            
            if len(combo_data) == 0:
                continue
            
            combo_results = combo_data.groupby('test_system_lower').mean(numeric_only=True)
            
            random_data = results_df[results_df['type'] == 'random']
            random_results = random_data.groupby('test_system_lower').mean(numeric_only=True)
            
            combo_latex = f"\\begin{{table}}[htbp]\n\\centering\n\\renewcommand{{\\arraystretch}}{{1.2}}\n"
            combo_latex += "\\begin{tabular}{l|cccc|cccc}\n\\hline\n"
            combo_latex += "System & NDCG@1 & NDCG@10 & NDCG@20 & NDCG(all) & AP@1 & AP@10 & AP@20 & MAP(all) \\\\ \\hline\n"
            
            pos_count = 0
            neg_count = 0
            
            for system in self.system_order:
                sys_key = system.lower()
                if sys_key not in combo_results.index or sys_key not in random_results.index:
                    continue
                    
                combo_latex += f"{self.system_latex_names[sys_key]} & "
                
                for idx, metric in enumerate(metrics):
                    model_vals = combo_data[combo_data['test_system_lower'] == sys_key][metric]
                    random_vals = random_data[random_data['test_system_lower'] == sys_key][metric]
                    
                    if len(model_vals) == 0 or len(random_vals) == 0:
                        continue
                        
                    model_mean = model_vals.mean()
                    random_mean = random_vals.mean()
                    imp = (model_mean / random_mean - 1) * 100
                    
                    if imp > 0:
                        pos_count += 1
                    elif imp < 0:
                        neg_count += 1
                    
                    try:
                        u_stat, p_value = mannwhitneyu(model_vals, random_vals, alternative='two-sided')
                    except Exception:
                        p_value = 1.0
                    
                    if p_value < 0.001:
                        sig = "$^\\dagger$"
                    elif p_value < 0.05:
                        sig = "$^\\star$"
                    else:
                        sig = "$^{\,\,\,}$"
                    
                    if imp > 0:
                        cell = f"\\cellcolor{{green!30}}{model_mean:.4f}\\textsuperscript{{+{imp:.1f}\\%}}{sig}"
                    else:
                        cell = f"\\cellcolor{{red!30}}{model_mean:.4f}\\textsuperscript{{{imp:.1f}\\%}}{sig}"
                    
                    if idx < len(metrics) - 1:
                        combo_latex += cell + " & "
                    else:
                        combo_latex += cell + " "
                
                combo_latex += "\\\\\n"
            
            combo_latex += "\\hline\n"
            combo_latex += "\\textbf{Average} & "
            
            for idx, metric in enumerate(metrics):
                model_avg = combo_results[metric].mean()
                random_avg = random_results[metric].mean()
                imp_avg = (model_avg / random_avg - 1) * 100
                
                all_model_vals = combo_data[metric]
                all_random_vals = random_data[metric]
                
                try:
                    u_stat, p_value = mannwhitneyu(all_model_vals, all_random_vals, alternative='two-sided')
                except Exception:
                    p_value = 1.0
                
                if p_value < 0.001:
                    sig = "$^\\dagger$"
                elif p_value < 0.05:
                    sig = "$^\\star$"
                else:
                    sig = "$^{\,\,\,}$"
                
                if imp_avg > 0:
                    cell = f"\\cellcolor{{green!30}}\\textbf{{{model_avg:.4f}}}\\textsuperscript{{+{imp_avg:.1f}\\%}}{sig}"
                else:
                    cell = f"\\cellcolor{{red!30}}\\textbf{{{model_avg:.4f}}}\\textsuperscript{{{imp_avg:.1f}\\%}}{sig}"
                
                if idx < len(metrics) - 1:
                    combo_latex += cell + " & "
                else:
                    combo_latex += cell + " "
            
            combo_latex += "\\\\\n"
            combo_latex += "\\hline\n"
            combo_latex += f"\\end{{tabular}}\n"
            combo_latex += f"\\caption{{Performance comparison for feature combination {combo} on batch data}}\n"
            combo_latex += f"\\label{{tab:combo_{combo.replace('+', '_')}_performance_batch}}\n\\end{{table}}"
            
            safe_combo_name = combo.replace('+', '_').replace('/', '_')
            with open(f'feature_combo_tables_batch/{safe_combo_name}_performance.tex', 'w', encoding='utf-8') as f:
                f.write(combo_latex)
            
            logger.info(f"batch feature combination table for {combo} saved to 'feature_combo_tables_batch/{safe_combo_name}_performance.tex'")
            logger.info(f"Combo {combo} - Positive improvements: {pos_count}, Negative improvements: {neg_count}")
    
    def generate_feature_combination_summary_table(self, results_df: pd.DataFrame) -> str:
        """Generate a summary LaTeX table comparing all feature combinations"""
        results_df = results_df.copy()
        results_df['type'] = results_df['feature_combo'].apply(lambda x: 'random' if x == 'random' else 'model')
        
        metrics = ['ndcg_1', 'ndcg_10', 'ndcg_20', 'ndcg_all', 'map_1', 'map_10', 'map_20', 'map_all']
        summary = results_df.groupby(['feature_combo', 'type'])[metrics].mean()
        
        feature_combos = results_df[results_df['type'] == 'model']['feature_combo'].unique()
        random_performance = summary.loc[('random', 'random')]
        
        summary_latex = "\\begin{table}[htbp]\n\\centering\n\\renewcommand{\\arraystretch}{1.2}\n"
        summary_latex += "\\begin{tabular}{l|cccc|cccc}\n\\hline\n"
        summary_latex += "Combination & NDCG@1 & NDCG@10 & NDCG@20 & NDCG(all) & AP@1 & AP@10 & AP@20 & MAP(all) \\\\ \\hline\n"
        
        for combo in sorted(feature_combos):
            combo_performance = summary.loc[(combo, 'model')]
            summary_latex += f"{combo} & "
            
            for idx, metric in enumerate(metrics):
                model_vals = results_df[(results_df['feature_combo'] == combo) & (results_df['type'] == 'model')][metric]
                random_vals = results_df[results_df['type'] == 'random'][metric]
                
                model_mean = model_vals.mean()
                random_mean = random_vals.mean()
                imp = (model_mean / random_mean - 1) * 100
                
                try:
                    u_stat, p_value = mannwhitneyu(model_vals, random_vals, alternative='two-sided')
                except Exception:
                    p_value = 1.0
                
                if p_value < 0.001:
                    sig = "$^\\dagger$"
                elif p_value < 0.05:
                    sig = "$^\\star$"
                else:
                    sig = "$^{\,\,\,}$"
                
                if imp > 0:
                    cell = f"\\cellcolor{{green!30}}{model_mean:.4f}\\textsuperscript{{+{imp:.1f}\\%}}{sig}"
                else:
                    cell = f"\\cellcolor{{red!30}}{model_mean:.4f}\\textsuperscript{{{imp:.1f}\\%}}{sig}"
                
                if idx < len(metrics) - 1:
                    summary_latex += cell + " & "
                else:
                    summary_latex += cell + " "
            
            summary_latex += "\\\\\n"
        
        summary_latex += "\\hline\n"
        summary_latex += "Random Baseline & "
        for idx, metric in enumerate(metrics):
            random_mean = random_performance[metric]
            cell = f"{random_mean:.4f}"
            if idx < len(metrics) - 1:
                summary_latex += cell + " & "
            else:
                summary_latex += cell + " "
        summary_latex += "\\\\\n"
        summary_latex += "\\hline\n\\end{tabular}\n"
        summary_latex += "\\caption{Summary of feature combination performance on batch data}\n"
        summary_latex += "\\label{tab:feature_combo_summary_batch}\n\\end{table}"
        
        return summary_latex
    
    def create_visualizations(self, summary: pd.DataFrame):
        """Create visualization plots"""
        self._create_heatmaps(summary)
        self._create_comparison_barplot(summary)
    
    def _create_heatmaps(self, summary: pd.DataFrame):
        """Create heatmap visualizations"""
        heatmap_data = summary[['ndcg_1', 'ndcg_all']].drop('random', errors='ignore')
        
        for metric in ['ndcg_1', 'ndcg_all']:
            heatmap_matrix = self._prepare_heatmap_data(heatmap_data, metric)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(heatmap_matrix, annot=True, fmt='.4f', cmap='viridis',
                       xticklabels=self.feature_manager.direct_metrics,
                       yticklabels=self.feature_manager.global_metrics)
            
            metric_title = 'NDCG@1' if metric == 'ndcg_1' else 'NDCG(all)'
            plt.title(f'{metric_title} Performance of Each Feature Combination (batch)', fontsize=14)
            plt.xlabel('Direct Metrics', fontsize=12)
            plt.ylabel('Global Metrics', fontsize=12)
            plt.tight_layout()


    
    def _prepare_heatmap_data(self, heatmap_data: pd.DataFrame, metric: str) -> np.ndarray:
        """Prepare data matrix for heatmap visualization"""
        global_names = self.feature_manager.global_metrics
        direct_names = self.feature_manager.direct_metrics
        
        heatmap_matrix = np.zeros((len(global_names), len(direct_names)))
        
        for g_idx, g_metric in enumerate(global_names):
            for d_idx, d_metric in enumerate(direct_names):
                combo_name = f"{g_metric}+{d_metric}"
                if combo_name in heatmap_data.index:
                    heatmap_matrix[g_idx, d_idx] = heatmap_data.loc[combo_name, metric]
        
        return heatmap_matrix
    
    def _create_comparison_barplot(self, summary: pd.DataFrame):
        """Create bar plot comparison"""
        bar_data = summary[['ndcg_1']].sort_values('ndcg_1', ascending=False)
        bar_data = bar_data.drop('random', errors='ignore')
        random_performance = summary.loc['random', 'ndcg_1']
        
        plt.figure(figsize=(14, 8))
        colors = sns.color_palette('viridis', len(bar_data))
        
        bars = plt.bar(bar_data.index, bar_data['ndcg_1'], color=colors)
        plt.axhline(y=random_performance, color='r', linestyle='--',
                   label=f'Random baseline ({random_performance:.4f})')
        
        plt.title('NDCG@1 Performance Comparison of Each Feature Combination (batch)', fontsize=14)
        plt.xlabel('Feature Combination', fontsize=12)
        plt.ylabel('NDCG@1', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()

    
    def print_best_combinations(self, summary: pd.DataFrame):
        """Print best performing feature combinations"""
        best_combos = {}
        metrics_of_interest = ['ndcg_1', 'ndcg_all', 'map_1']
        
        for metric in metrics_of_interest:
            best_combo = summary[metric].drop('random', errors='ignore').idxmax()
            best_score = summary.loc[best_combo, metric]
            best_combos[metric] = (best_combo, best_score)
        
        print("\n===== Best Feature Combinations (batch) =====")
        for metric, (combo, score) in best_combos.items():
            metric_name = metric.upper().replace('_', '@')
            print(f"{metric_name} best combination: {combo} (score: {score:.4f})")


class Model4TuneBatchExperiment:
    """Main experiment orchestrator for batch data"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.feature_manager = FeatureManager()
        self.evaluator = ModelEvaluator(config)
        self.analyzer = ResultsAnalyzer(self.feature_manager)
    
    def run_experiment(self):
        """Run the complete experiment"""
        logger.info("Starting Model4Tune Batch experiment...")
        logger.info(f"Using learning rate range: {self.config.base_learning_rate:.3f} to {self.config.base_learning_rate + (self.config.n_iterations-1) * self.config.learning_rate_step:.3f}")
        
        # Load and prepare data
        df = pd.read_csv(self.config.data_path)
        y = df['perf'].values
        systems = df['System'].values
        
        # Convert to rankings
        ranking_utils = RankingUtilities()
        y_rank = ranking_utils.convert_to_rank_per_system(y, systems, self.config.max_rank)
        
        # Print dataset information
        self._print_dataset_info(df, systems, y_rank)
        
        # Generate feature combinations
        feature_combinations = self.feature_manager.generate_feature_combinations()
        logger.info(f"Evaluating {len(feature_combinations)} feature combinations...")
        
        # Run experiments
        all_results = []
        
        # Evaluate feature combinations
        for idx, (features, combo_name) in enumerate(feature_combinations):
            logger.info(f"Evaluating combination {idx+1}/{len(feature_combinations)}: {combo_name} (features: {len(features)})")
            
            X = df[features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            results = self.evaluator.evaluate_single_combination(X_scaled, y_rank, systems, combo_name)
            all_results.extend(results)
            logger.info(f"Completed combination {combo_name}, collected {len(results)} results")
        
        # Evaluate random baseline
        random_results = self.evaluator.evaluate_random_baseline(y_rank, systems)
        all_results.extend(random_results)
        
        # Analyze results
        results_df = pd.DataFrame(all_results)
        summary, improvement = self.analyzer.analyze_results(results_df)
        
        # Save results
        # self._save_results(results_df, summary, improvement)
        
        # Print analysis (now includes system performance table)
        self._print_analysis(summary, improvement, results_df)
        
        logger.info("Batch experiment completed successfully!")
    
    def _print_dataset_info(self, df: pd.DataFrame, systems: np.ndarray, y_rank: np.ndarray):
        """Print dataset information"""
        print(f"Total samples: {len(df)}")
        print(f"Total systems: {len(np.unique(systems))}")
        print(f"Ranking range: [{y_rank.min()}, {y_rank.max()}]")
    
    def _save_results(self, results_df: pd.DataFrame, summary: pd.DataFrame, improvement: pd.DataFrame):
        """Save experiment results"""
        results_df.to_csv('feature_combination_results.csv', index=False)
        summary.to_csv('feature_combination_summary.csv')
        improvement.to_csv('feature_combination_improvement.csv')
        
        logger.info("Results saved to CSV files")
    
    def _print_analysis(self, summary: pd.DataFrame, improvement: pd.DataFrame, results_df: pd.DataFrame):
        """Print analysis results"""
        print("\n===== Performance comparison of each feature combination =====")
        print(summary)
        
        print("\n===== Performance improvement percentage relative to random =====")
        print(improvement[improvement.index != 'random'].round(2))
        
        self.analyzer.print_best_combinations(summary)
        
        print("\n===== Generating feature combination summary LaTeX table =====")
        summary_latex = self.analyzer.generate_feature_combination_summary_table(results_df)
        print(summary_latex)
        
        
        print("\n===== Generating individual feature combination tables =====")
        self.analyzer.generate_feature_combination_tables(results_df)


def main():
    """Main execution function"""
    config = ExperimentConfig(
        base_learning_rate=0.05,    
        learning_rate_step=0.01,      
        n_iterations=5               
    )
    experiment = Model4TuneBatchExperiment(config)
    experiment.run_experiment()


if __name__ == "__main__":
    main()