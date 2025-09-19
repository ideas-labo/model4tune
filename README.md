# Model4Tune

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

This repository contains the implementation and experimental code for 
> unveiling Many Faces of Surrogate Model for Configuration Tuning: A Fitness Landscape Analysis Perspective

## Introduction
```
To efficiently tune configuration for better system performance (e.g., latency), many tuners have leveraged a surrogate model to expedite the process instead of solely relying on the profoundly expensive system measurement. As such, it is naturally believe that we need more accurate model. However, the fact of "accuracy can lie"-a somewhat surprising finding from prior work-has left us many unanswered questions regarding what role does the surrogate model plays for configuration tuning. This paper provides the very first systematic exploration and discussion, together with a resolution proposal, to disclose the many faces of surrogate model for configuration tuning, through the novel perspective of fitness landscape analysis. We propose a theory as an alternative to accuracy for assessing the model usefulness for configuration, based on which we conduct an extensive empirical studying involving about 28,980 cases. Drawing on the above, we propose Model4Tune, an automated predictive tool that estimates which model-tuner pairs are the best for an unforeseen system without expensive profiling. Our results suggest that Model4Tune, as one of the first of its kind, performs 92.1% better than random guessing. Our work does not only shed lights on the possible future research directions, but also offers a practical resolution that can assist practitioners in evaluating the most useful model for configuration tuning.
```
## Key Features

- **Real Space Landscape Measurement**: Compute landscape metrics (FDC, PLO, NBC, etc.) from actual performance measurements
- **Model-based Landscape Analysis**: Evaluate landscapes predicted by surrogate models
- **Batch and Sequential Experiments**: Comprehensive experiment frameworks for different sampling strategies
- **Research Question Analysis**: Scripts for reproducing specific research questions from the paper

## Repository Structure

```
Model4Tune/
├── README.md
├── requirements.txt
├── Landscape_utils.py              # Core landscape computation utilities
├── Measure_Landscape_real_space.py # Real space landscape measurement
├── Measure_Landscape_model_*.py    # Model-based landscape analysis
├── Model4Tune_batch.py            # Batch predict framework
├── Model4Tune_sequential.py       # Sequential predict framework
├── RQ*.py                         # Research question analysis scripts
├── utils/                         # Utility modules
├── Data/                          # Input datasets
├── landscape_data/                # Landscape measurement results
├── processed_data/                # Processed landscape results
├── results/                       # Experiment outputs
└── feature_combo_tables/          # Generated LaTeX tables
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Required packages listed in `requirements.txt`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Model4Tune.git
cd Model4Tune
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

1. **Measure real space landscapes**:
```bash
python Measure_Landscape_real_space.py
```

2. **Analyze model-predicted landscapes**:
```bash
python Measure_Landscape_model_for_all_options.py
```

3. **Run batch prediction**:
```bash
python Model4Tune_batch.py
```

4. **Run sequential prediction**:
```bash
python Model4Tune_sequential.py
```

## Reproducing Paper Results

To reproduce the results from our paper, run the following scripts in order:

1. **RQ1 - Theory Validation**:
```bash
python RQ1_test_the_theory.py
```

2. **RQ2 - Accuracy-Landscape Correlation**:
```bash
python RQ2_acc_landscape_correlation.py
```

3. **RQ3 - Model Fidelity and Best Model**:
```bash
python RQ3_fidelity.py
python RQ3_best_model.py
```

4. **RQ4 - Individual Option Influence**:
```bash
python RQ4_individual_option_influence.py
```

## Tuner

Specification can be seen in [Tunres](https://github.com/ideas-labo/model-impact).


## Data Format

The framework expects performance data in CSV format with:
- Feature columns (binary or numerical)
- Performance column (continuous values)
- Optional metadata columns

Example:
```csv
feature1,feature2,feature3,performance
0,1,0,145.2
1,0,1,198.7
...
```

<!-- ## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{model4tune2024,
  title={Model4Tune: Performance Prediction and Landscape Analysis for Software Product Lines},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2024},
  publisher={[Publisher]}
}
``` -->

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




