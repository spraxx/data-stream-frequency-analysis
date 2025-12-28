"""Frequent items analysis in data streams using exact, approximate, and Space-Saving algorithms."""

import os
import random
import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class FrequentItemsProject:
    """Analyzes frequent items in data streams using multiple counting methods.
    
    Implements three approaches:
    - Exact counters (baseline)
    - Approximate counters with fixed probability
    - Space-Saving algorithm for top-k estimation
    
    Attributes:
        df: DataFrame containing the dataset
        data_stream: List of items to analyze
        n_total: Total number of items in stream
    """
    
    def __init__(self, file_path: str) -> None:
        """Initialize project by loading dataset.
        
        Args:
            file_path: Path to CSV file containing dataset
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If 'release_year' column is missing
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"CSV file not found: {file_path}\n"
                f"Please ensure the file exists at the specified path."
            )
        
        self.df = pd.read_csv(file_path)
        
        if 'release_year' not in self.df.columns:
            raise ValueError(
                f"Column 'release_year' not found in dataset.\n"
                f"Available columns: {list(self.df.columns)}"
            )
        
        self.data_stream = self.df['release_year'].tolist()
        self.n_total = len(self.data_stream)
        print(f"Dataset loaded: {self.n_total} records, "
              f"{len(set(self.data_stream))} unique items\n")

    def get_exact_counts(self) -> Counter:
        """Compute exact count of each item using Counter.
        
        Returns:
            Counter object with exact item frequencies
        """
        return Counter(self.data_stream)

    def fixed_probability_counter(self, p: float = 0.5) -> Dict[int, int]:
        """Estimate item counts using fixed probability sampling.
        
        Args:
            p: Probability of sampling each item (default: 0.5)
            
        Returns:
            Dictionary mapping items to estimated counts
        """
        counts = {}
        for item in self.data_stream:
            if random.random() < p:
                counts[item] = counts.get(item, 0) + 1
        
        # Scale up by 1/p to estimate total counts
        return {k: int(v / p) for k, v in counts.items()}

    def space_saving(self, m: int) -> Dict[int, int]:
        """Find frequent items using Space-Saving algorithm.
        
        Maintains exactly m counters. When full, replaces minimum counter
        with new item, incrementing by the replaced count.
        
        Args:
            m: Number of counters to maintain
            
        Returns:
            Dictionary mapping items to estimated counts
        """
        counters = {}
        for item in self.data_stream:
            if item in counters:
                counters[item] += 1
            elif len(counters) < m:
                counters[item] = 1
            else:
                min_item = min(counters, key=counters.get)
                min_val = counters[min_item]
                del counters[min_item]
                counters[item] = min_val + 1
        return counters

    def run_full_analysis(self) -> None:
        """Execute complete analysis and save results to CSV files.
        
        Generates:
            - results/approximate_counter_stats.csv: Error statistics
            - results/approximate_counter_top10.csv: Top 10 comparison
            - results/space_saving_performance.csv: Precision/recall metrics
            - results/space_saving_top10.csv: Top 10 comparison
            - results/order_preservation.csv: Order preservation metrics
            - results/memory_growth.csv: Memory growth over time
            - results/probability_tradeoff.csv: Probability vs accuracy tradeoff
            - results/execution_times.csv: Algorithm execution time comparison
            - results/least_frequent_analysis.csv: Bottom-5 items comparison
        """
        os.makedirs('results', exist_ok=True)
        
        exact = self.get_exact_counts()
        exact_sorted = sorted(exact.items(), key=lambda x: x[1], reverse=True)
        
        print("=" * 70)
        print("RUNNING ANALYSIS")
        print("=" * 70)
        
        # Approximate counter analysis
        print("\n[1/7] Analyzing approximate counter (10 trials)...")
        approx_stats, approx_top10 = self._analyze_approximate_counter(exact, exact_sorted)
        
        # Space-Saving analysis
        print("[2/7] Analyzing Space-Saving algorithm...")
        ss_performance, ss_top10 = self._analyze_space_saving(exact, exact_sorted)
        
        # Order preservation analysis
        print("[3/7] Analyzing order preservation...")
        order_stats = self._analyze_order_preservation(exact_sorted)
        
        # Memory growth analysis
        print("[4/7] Analyzing memory growth over time...")
        memory_growth = self._analyze_memory_growth()
        
        # Probability trade-off analysis
        print("[5/7] Analyzing probability trade-off (multiple p values)...")
        prob_tradeoff = self._analyze_probability_tradeoff()
        
        # Execution time analysis
        print("[6/7] Measuring execution times...")
        exec_times = self._analyze_execution_times()
        
        # Least frequent analysis
        print("[7/7] Analyzing least frequent items...")
        least_freq = self._analyze_least_frequent(exact, exact_sorted)
        
        # Save all results
        approx_stats.to_csv('results/approximate_counter_stats.csv', index=False)
        approx_top10.to_csv('results/approximate_counter_top10.csv', index=False)
        ss_performance.to_csv('results/space_saving_performance.csv', index=False)
        ss_top10.to_csv('results/space_saving_top10.csv', index=False)
        order_stats.to_csv('results/order_preservation.csv', index=False)
        memory_growth.to_csv('results/memory_growth.csv', index=False)
        prob_tradeoff.to_csv('results/probability_tradeoff.csv', index=False)
        exec_times.to_csv('results/execution_times.csv', index=False)
        least_freq.to_csv('results/least_frequent_analysis.csv', index=False)
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print("\nResults saved to 'results/' directory:")
        print("  - approximate_counter_stats.csv")
        print("  - approximate_counter_top10.csv")
        print("  - space_saving_performance.csv")
        print("  - space_saving_top10.csv")
        print("  - order_preservation.csv")
        print("  - memory_growth.csv")
        print("  - probability_tradeoff.csv")
        print("  - execution_times.csv")
        print("  - least_frequent_analysis.csv")
        print("=" * 70)
    
    def _analyze_approximate_counter(
        self, 
        exact: Counter, 
        exact_sorted: List[Tuple[int, int]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Analyze approximate counter performance.
        
        Args:
            exact: Exact counts
            exact_sorted: Sorted list of (item, count) tuples
            
        Returns:
            Tuple of (statistics DataFrame, top10 comparison DataFrame)
        """
        num_trials = 10
        trial_abs_errors = []
        trial_rel_errors = []
        trial_missing_items = []
        
        for _ in range(num_trials):
            approx = self.fixed_probability_counter()
            abs_errors = []
            rel_errors = []
            missing_count = 0
            
            for year, actual_count in exact.items():
                est_count = approx.get(year, 0)
                abs_err = abs(actual_count - est_count)
                abs_errors.append(abs_err)
                rel_errors.append(abs_err / actual_count if actual_count > 0 else 0)
                
                if est_count == 0:
                    missing_count += 1
            
            trial_abs_errors.append(abs_errors)
            trial_rel_errors.append(rel_errors)
            trial_missing_items.append(missing_count)
        
        all_abs = [err for trial in trial_abs_errors for err in trial]
        all_rel = [err for trial in trial_rel_errors for err in trial]
        
        # Statistics DataFrame
        stats_df = pd.DataFrame({
            'metric': [
                'absolute_error_mean', 'absolute_error_min', 
                'absolute_error_max', 'absolute_error_std',
                'relative_error_mean_pct', 'relative_error_min_pct',
                'relative_error_max_pct', 'relative_error_std_pct',
                'missing_items_mean', 'missing_items_min',
                'missing_items_max', 'missing_items_pct'
            ],
            'value': [
                np.mean(all_abs), np.min(all_abs),
                np.max(all_abs), np.std(all_abs),
                np.mean(all_rel) * 100, np.min(all_rel) * 100,
                np.max(all_rel) * 100, np.std(all_rel) * 100,
                np.mean(trial_missing_items), np.min(trial_missing_items),
                np.max(trial_missing_items), 
                np.mean(trial_missing_items) / len(exact) * 100
            ]
        })
        
        # Top 10 comparison
        approx_example = self.fixed_probability_counter()
        top10_data = []
        for rank, (year, exact_count) in enumerate(exact_sorted[:10], 1):
            approx_count = approx_example.get(year, 0)
            abs_err = abs(exact_count - approx_count)
            rel_err = abs_err / exact_count if exact_count > 0 else 0
            
            top10_data.append({
                'rank': rank,
                'year': year,
                'exact_count': exact_count,
                'approx_count': approx_count,
                'abs_error': abs_err,
                'rel_error_pct': rel_err * 100
            })
        
        top10_df = pd.DataFrame(top10_data)
        
        return stats_df, top10_df
    
    def _analyze_space_saving(
        self,
        exact: Counter,
        exact_sorted: List[Tuple[int, int]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Analyze Space-Saving algorithm performance.
        
        Args:
            exact: Exact counts
            exact_sorted: Sorted list of (item, count) tuples
            
        Returns:
            Tuple of (performance metrics DataFrame, top10 comparison DataFrame)
        """
        # Extended range to show convergence toward exact counts
        n_values = [5, 10, 15, 20, 30, 50, 90]
        performance_data = []
        
        for n in n_values:
            ss_results = self.space_saving(n)
            ss_sorted = sorted(ss_results.items(), key=lambda x: x[1], reverse=True)[:n]
            
            top_exact_set = set([item[0] for item in exact_sorted[:n]])
            top_ss_set = set([item[0] for item in ss_sorted])
            
            true_positives = len(top_exact_set.intersection(top_ss_set))
            
            precision = true_positives / len(top_ss_set) if len(top_ss_set) > 0 else 0
            recall = true_positives / len(top_exact_set) if len(top_exact_set) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            performance_data.append({
                'top_k': n,
                'num_counters': n,
                'precision_pct': precision * 100,
                'recall_pct': recall * 100,
                'f1_score_pct': f1 * 100,
                'matches': true_positives,
                'total': n
            })
        
        performance_df = pd.DataFrame(performance_data)
        
        # Top 10 comparison with m=10
        ss_10 = self.space_saving(10)
        ss_dict = dict(sorted(ss_10.items(), key=lambda x: x[1], reverse=True))
        
        top10_data = []
        for rank, (year, exact_count) in enumerate(exact_sorted[:10], 1):
            ss_count = ss_dict.get(year, 0)
            match = ss_count > 0
            
            top10_data.append({
                'rank': rank,
                'year': year,
                'exact_count': exact_count,
                'ss_count': ss_count,
                'match': match
            })
        
        top10_df = pd.DataFrame(top10_data)
        
        return performance_df, top10_df
    
    def _analyze_order_preservation(
        self,
        exact_sorted: List[Tuple[int, int]]
    ) -> pd.DataFrame:
        """Analyze order preservation in Space-Saving algorithm.
        
        Args:
            exact_sorted: Sorted list of (item, count) tuples
            
        Returns:
            DataFrame with order preservation metrics
        """
        order_data = []
        
        for n in [5, 10]:
            exact_order = [x[0] for x in exact_sorted[:n]]
            ss_n = self.space_saving(n * 2)
            ss_order = [x[0] for x in sorted(ss_n.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True)[:n]]
            
            order_matches = sum(1 for i, item in enumerate(exact_order) 
                              if i < len(ss_order) and item == ss_order[i])
            
            order_data.append({
                'top_k': n,
                'num_counters': n * 2,
                'positions_preserved': order_matches,
                'total_positions': n,
                'preservation_pct': order_matches / n * 100,
                'exact_order': str(exact_order),
                'ss_order': str(ss_order)
            })
        
        return pd.DataFrame(order_data)
    
    def _analyze_memory_growth(self) -> pd.DataFrame:
        """Track memory growth over time for exact and approximate counters.
        
        Returns:
            DataFrame with time-series memory usage data
        """
        memory_data = []
        
        # Exact counter tracking
        exact_seen = set()
        
        # Fixed probability counter tracking (p=0.5)
        p = 0.5
        approx_seen = set()
        
        for i, item in enumerate(self.data_stream, 1):
            # Track exact counter
            exact_seen.add(item)
            
            # Track approximate counter (sample with probability p)
            if random.random() < p:
                approx_seen.add(item)
            
            # Record at intervals
            if i % 50 == 0 or i == len(self.data_stream):
                memory_data.append({
                    'items_processed': i,
                    'exact_counters': len(exact_seen),
                    'approx_counters': len(approx_seen),
                    'approx_percentage': len(approx_seen) / len(exact_seen) * 100 if len(exact_seen) > 0 else 0
                })
        
        return pd.DataFrame(memory_data)
    
    def _analyze_probability_tradeoff(self) -> pd.DataFrame:
        """Analyze trade-off between memory and accuracy across different p values.
        
        Returns:
            DataFrame with memory usage and error metrics for different p values
        """
        p_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        exact = self.get_exact_counts()
        tradeoff_data = []
        
        for p in p_values:
            # Run multiple trials for each p value
            num_trials = 5
            trial_memory = []
            trial_abs_errors = []
            trial_rel_errors = []
            trial_missing = []
            
            for _ in range(num_trials):
                approx = self.fixed_probability_counter(p)
                
                # Track memory (unique items sampled)
                trial_memory.append(len(approx))
                
                # Calculate errors
                abs_errors = []
                rel_errors = []
                missing_count = 0
                
                for year, actual_count in exact.items():
                    est_count = approx.get(year, 0)
                    abs_err = abs(actual_count - est_count)
                    abs_errors.append(abs_err)
                    rel_errors.append(abs_err / actual_count if actual_count > 0 else 0)
                    
                    if est_count == 0:
                        missing_count += 1
                
                trial_abs_errors.append(np.mean(abs_errors))
                trial_rel_errors.append(np.mean(rel_errors))
                trial_missing.append(missing_count)
            
            # Average across trials
            tradeoff_data.append({
                'probability_p': p,
                'memory_counters': np.mean(trial_memory),
                'memory_percentage': np.mean(trial_memory) / len(exact) * 100,
                'mean_abs_error': np.mean(trial_abs_errors),
                'std_abs_error': np.std(trial_abs_errors),
                'mean_rel_error_pct': np.mean(trial_rel_errors) * 100,
                'std_rel_error_pct': np.std(trial_rel_errors) * 100,
                'missing_items': np.mean(trial_missing),
                'missing_items_pct': np.mean(trial_missing) / len(exact) * 100
            })
        
        return pd.DataFrame(tradeoff_data)
    
    def _analyze_execution_times(self) -> pd.DataFrame:
        """Measure execution time for each algorithm processing 1,450 records.
        
        Returns:
            DataFrame with execution times in milliseconds
        """
        time_data = []
        num_trials = 10
        
        # Exact counter timing
        exact_times = []
        for _ in range(num_trials):
            start = time.perf_counter()
            self.get_exact_counts()
            exact_times.append((time.perf_counter() - start) * 1000)
        
        time_data.append({
            'algorithm': 'Exact Counter',
            'mean_time_ms': np.mean(exact_times),
            'std_time_ms': np.std(exact_times),
            'min_time_ms': np.min(exact_times),
            'max_time_ms': np.max(exact_times)
        })
        
        # Approximate counter timing (p=0.5)
        approx_times = []
        for _ in range(num_trials):
            start = time.perf_counter()
            self.fixed_probability_counter(0.5)
            approx_times.append((time.perf_counter() - start) * 1000)
        
        time_data.append({
            'algorithm': 'Approximate Counter (p=0.5)',
            'mean_time_ms': np.mean(approx_times),
            'std_time_ms': np.std(approx_times),
            'min_time_ms': np.min(approx_times),
            'max_time_ms': np.max(approx_times)
        })
        
        # Space-Saving timing for different m values
        for m in [5, 10, 15, 20, 30, 50, 90]:
            ss_times = []
            for _ in range(num_trials):
                start = time.perf_counter()
                self.space_saving(m)
                ss_times.append((time.perf_counter() - start) * 1000)
            
            time_data.append({
                'algorithm': f'Space-Saving (m={m})',
                'mean_time_ms': np.mean(ss_times),
                'std_time_ms': np.std(ss_times),
                'min_time_ms': np.min(ss_times),
                'max_time_ms': np.max(ss_times)
            })
        
        return pd.DataFrame(time_data)
    
    def _analyze_least_frequent(
        self,
        exact: Counter,
        exact_sorted: List[Tuple[int, int]]
    ) -> pd.DataFrame:
        """Analyze how algorithms handle the least frequent items.
        
        Compares bottom-5 items from exact counter against approximate
        and Space-Saving algorithms to evaluate performance on rare items.
        
        Args:
            exact: Exact counts
            exact_sorted: Sorted list of (item, count) tuples
            
        Returns:
            DataFrame comparing bottom-5 items across algorithms
        """
        # Get bottom 5 items from exact counter
        bottom5_exact = sorted(exact.items(), key=lambda x: x[1])[:5]
        
        # Run algorithms
        approx = self.fixed_probability_counter(0.5)
        ss_10 = self.space_saving(10)
        ss_20 = self.space_saving(20)
        ss_50 = self.space_saving(50)
        ss_90 = self.space_saving(90)
        
        least_freq_data = []
        for rank, (year, exact_count) in enumerate(bottom5_exact, 1):
            least_freq_data.append({
                'rank': rank,
                'year': year,
                'exact_count': exact_count,
                'approx_count': approx.get(year, 0),
                'approx_found': year in approx,
                'ss_10_count': ss_10.get(year, 0),
                'ss_10_found': year in ss_10,
                'ss_20_count': ss_20.get(year, 0),
                'ss_20_found': year in ss_20,
                'ss_50_count': ss_50.get(year, 0),
                'ss_50_found': year in ss_50,
                'ss_90_count': ss_90.get(year, 0),
                'ss_90_found': year in ss_90
            })
        
        return pd.DataFrame(least_freq_data)

if __name__ == "__main__":
    try:
        project = FrequentItemsProject('data/disney_plus_titles.csv')
        project.run_full_analysis()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")