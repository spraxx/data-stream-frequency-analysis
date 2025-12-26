# Frequent Items Analysis in Data Streams

A Python implementation comparing three algorithms for identifying frequent items in data streams: exact counters, approximate counters with fixed probability sampling, and the Space-Saving algorithm.

## Overview

This project analyzes the `release_year` attribute from a Disney+ titles dataset (1,450 records, 90 unique years) to evaluate the trade-offs between accuracy, memory usage, and computational efficiency in streaming data analysis.

## Algorithms Implemented

### 1. Exact Counters (Baseline)
- **Method**: Python's `Counter` from collections module
- **Space Complexity**: O(n) where n = number of unique items
- **Accuracy**: 100%
- **Use Case**: Ground truth for comparison

### 2. Fixed Probability Counter
- **Method**: Samples each item with probability p = 0.5
- **Space Complexity**: ~50% of exact counter space
- **Accuracy**: ~67% (avg. 33% relative error)
- **Characteristics**:
  - Unbiased estimator (scales counts by 1/p)
  - Probabilistic - results vary across runs
  - May miss ~7% of items (never sampled)

### 3. Space-Saving Algorithm
- **Method**: Maintains exactly m counters for top-k estimation
- **Space Complexity**: O(m) - bounded memory
- **Accuracy**: Depends on m/unique-items ratio
- **Characteristics**:
  - Deterministic
  - Best for m ≥ 15 counters (67% precision/recall on this dataset)
  - Performance degrades when m << unique items

## Setup

### Requirements
- Python 3.11+
- UV package manager

### Installation

```bash
# Clone or download the project
cd data-stream-frequency-analysis

# Install dependencies using UV
uv sync
```

### Dependencies
- pandas >= 2.3.3
- numpy >= 2.4.0

## Usage

```bash
uv run python frequent_items.py
```

### Expected Output
```
Dataset loaded: 1450 records, 90 unique items

======================================================================
RUNNING ANALYSIS
======================================================================

[1/3] Analyzing approximate counter (10 trials)...
[2/3] Analyzing Space-Saving algorithm...
[3/3] Analyzing order preservation...

======================================================================
ANALYSIS COMPLETE
======================================================================

Results saved to 'results/' directory:
  - approximate_counter_stats.csv
  - approximate_counter_top10.csv
  - space_saving_performance.csv
  - space_saving_top10.csv
  - order_preservation.csv
```

## Results

All analysis results are exported to CSV files in the `results/` directory:

### 1. `approximate_counter_stats.csv`
Error metrics across 10 trials:
- Absolute error (mean, min, max, std)
- Relative error percentage (mean, min, max, std)
- Missing items statistics (mean, min, max, percentage)

### 2. `approximate_counter_top10.csv`
Top 10 items comparison:
- Rank, year, exact count, approximate count
- Absolute error, relative error percentage

### 3. `space_saving_performance.csv`
Algorithm performance by k-value (5, 10, 15, 20):
- Precision, recall, F1-score
- Number of matches vs total

### 4. `space_saving_top10.csv`
Top 10 items with Space-Saving (m=10):
- Rank, year, exact count, Space-Saving count
- Boolean match indicator

### 5. `order_preservation.csv`
Positional accuracy analysis:
- Top-5 and Top-10 order preservation
- Number of positions preserved
- Exact vs Space-Saving ordering

## Key Findings

Based on the Disney+ dataset analysis:

| Method | Space Usage | Accuracy | Deterministic | Missing Items |
|--------|-------------|----------|---------------|---------------|
| **Exact** | 90 counters | 100% | ✓ | 0% |
| **Approximate (p=0.5)** | ~45 counters | 67% avg | ✗ | ~7% |
| **Space-Saving (m=15)** | 15 counters | 67% | ✓ | 33% |

### Trade-offs
- **Exact**: Perfect accuracy but high memory cost
- **Approximate**: 50% space savings, probabilistic with variable accuracy
- **Space-Saving**: Fixed memory budget, accuracy scales with m

## Project Structure

```
data-stream-frequency-analysis/
├── data/
│   └── disney_plus_titles.csv    # Dataset (1,450 records)
├── results/                       # Generated CSV outputs
│   ├── approximate_counter_stats.csv
│   ├── approximate_counter_top10.csv
│   ├── space_saving_performance.csv
│   ├── space_saving_top10.csv
│   └── order_preservation.csv
├── frequent_items.py              # Main implementation
├── pyproject.toml                 # Dependencies
├── .python-version                # Python 3.11
└── README.md                      # This file
```

## Implementation Details

### Code Quality
- **Type annotations**: All functions use PEP 484 type hints
- **Docstrings**: Google-style documentation for all methods
- **PEP 8 compliant**: Proper formatting and style conventions
- **Error handling**: Graceful handling of missing files and columns

### Class Structure
```python
class FrequentItemsProject:
    def get_exact_counts() -> Counter
    def fixed_probability_counter(p: float = 0.5) -> Dict[int, int]
    def space_saving(m: int) -> Dict[int, int]
    def run_full_analysis() -> None
```

## Academic Context

This implementation addresses common challenges in streaming data analysis:

1. **Memory Constraints**: Real-world streams may have millions of unique items
2. **Accuracy vs Resources**: Trade-off between precision and computational cost
3. **Algorithm Selection**: Choosing appropriate method based on requirements

### Performance Metrics
- **Precision**: % of predicted top-k items that are correct
- **Recall**: % of actual top-k items that were found
- **F1-score**: Harmonic mean of precision and recall
- **Order preservation**: Positional accuracy in rankings

## License

This project is for educational purposes.

## Author

Data Streams Analysis Project