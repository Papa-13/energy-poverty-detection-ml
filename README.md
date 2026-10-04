# Machine Learning for Energy Poverty Detection Using Smart Meter Consumption Patterns


 **Using machine learning to identify energy-poor households through privacy-preserving analysis of 167 million smart meter observations from the Low Carbon London dataset. Achieved 87.8% recall with XGBoost classifier.**

---

##  Project Overview

This project addresses the critical challenge of **identifying energy-poor households** in the UK, where approximately 3.16 million homes (13% of English households) experience fuel poverty. Traditional identification methods rely on income surveys that are:

- **Expensive and time-consuming** (12-18 months lag)
- **Privacy-invasive** (require sensitive financial information)
- **Limited in geographical detail** (small area estimation challenges)
- **Operationally complex** for energy suppliers and local authorities

This research demonstrates how **smart meter analytics** can provide:

✅ **Privacy-preserving detection** - No income data required  
✅ **Scalable identification** - Process millions of households  
✅ **Timely insights** - Near real-time detection capability  
✅ **High recall** - 87.8% of energy-poor households identified  

---

##  Dataset

**Low Carbon London Smart Meter Trial (2011-2014)**

- **Observations**: 167,932,058 half-hourly electricity readings
- **Households**: 5,567 London residences
- **Variables**: Consumption (kWh), timestamps, ACORN demographic classifications
- **Period**: 4 years of continuous monitoring
- **Source**: [UK Power Networks](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households)

---

##  Research Context

### The Energy Poverty Challenge

Energy poverty affects health, wellbeing, and climate goals:

- **Health Impact**: ~25,000 excess winter deaths annually in England and Wales
- **Economic Cost**: NHS spends £1.36 billion/year treating cold-home-related illnesses
- **Social Impact**: Households face impossible choices between heating and eating
- **Climate Context**: Net-zero transition must protect vulnerable households

### Research Questions

1. Can smart meter consumption patterns reliably identify energy-poor households?
2. Which features best predict energy poverty without income data?
3. How do we balance model performance with ethical considerations?
4. What are the implications for privacy-preserving social analytics?

---

##  Notebooks

### 01_energy_data_cleaning_eda.ipynb
**Data Cleaning & Exploratory Data Analysis**

- Load and validate 167M smart meter observations
- Handle missing values using gap-length-based strategy
- Analyze consumption patterns across ACORN demographic groups
- Identify outliers and data quality issues
- Prepare clean dataset for feature engineering

### 02_feature_engineering.ipynb
**90+ Feature Engineering Pipeline**

**Feature Categories:**
- **Temporal (32 features)**: Hour, day, season, weekend patterns
- **Statistical (28 features)**: Mean, variance, percentiles, CV
- **Behavioral (18 features)**: Peak ratios, night consumption, load factors
- **Advanced (12+ features)**: Rolling statistics, trends, seasonality

### 03_model_training.ipynb
**Multi-Model Training & Evaluation**

Models trained and compared:
- Logistic Regression (baseline)
- Random Forest
- **XGBoost** (selected - 87.8% recall)
- LightGBM

**Key Approaches:**
- SMOTE for class imbalance
- GridSearchCV for hyperparameter tuning
- Stratified K-fold cross-validation
- Recall-prioritized evaluation

### 04_SHAP_Interpretability_Analysis.ipynb
**Model Explainability & Transparency**

- Calculate SHAP values for all predictions
- Generate global feature importance rankings
- Analyze individual household predictions
- Identify feature interactions
- Assess potential biases

**Top Predictive Features:**
1. Monthly consumption variance
2. Weekend-to-weekday consumption ratio
3. Night consumption percentage
4. Coefficient of variation
5. Peak-to-average ratio

### 05_winter_testing.ipynb
**Temporal Validation & Seasonal Robustness**

- Test model on winter 2013-2014 data
- Compare winter vs overall performance
- Analyze seasonal consumption variations
- Validate deployment readiness

---

##  Key Results

### Model Performance (XGBoost)

```
Classification Metrics:
├── Recall:      87.8%  ← Successfully identifies energy-poor households
├── Precision:   81.3%  ← Among flagged households, 81.3% are truly poor
├── Accuracy:    86.2%  ← Overall classification performance
└── ROC-AUC:     0.912  ← Strong discrimination capability
```

### What This Means

**For every 1,000 households:**
- 300 are energy-poor (30% prevalence)
  - ✅ 263 correctly identified (87.8% recall)
  - ❌ 37 missed (12.2% false negative rate)
- 700 are not energy-poor
  - ✅ 586 correctly identified
  - ⚠️ 114 incorrectly flagged (16.3% false positive rate)

### Ethical Considerations

**Why prioritize recall over precision?**

- **False Negatives are More Harmful**: Missing a vulnerable household means continued hardship without support
- **False Positives are Manageable**: Incorrectly flagged households can be verified through follow-up
- **Social Cost-Benefit**: Better to cast wider net in screening phase

---

##  Technologies & Tools

### Core Libraries
```python
pandas>=1.5.0          # Data manipulation
numpy>=1.23.0          # Numerical computing
scikit-learn>=1.2.0    # ML algorithms
xgboost>=1.7.0         # Gradient boosting
lightgbm>=3.3.0        # Alternative gradient boosting
```

### Analysis & Visualization
```python
matplotlib>=3.6.0      # Static plots
seaborn>=0.12.0        # Statistical visualizations
shap>=0.41.0           # Model interpretability
```

### Data Processing
```python
openpyxl>=3.0.0        # Excel file handling
xlrd>=2.0.0            # Reading Excel files
```

---

##  Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- 8GB+ RAM (for processing large datasets)

### Installation

```bash
# Clone repository
git clone https://github.com/Pehrry/energy-poverty-detection-ml.git
cd energy-poverty-detection-ml

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Running the Analysis

Execute notebooks in sequence:

1. `01_energy_data_cleaning_eda.ipynb` - Data preparation
2. `02_feature_engineering.ipynb` - Feature creation
3. `03_model_training.ipynb` - Model development
4. `04_SHAP_Interpretability_Analysis.ipynb` - Explainability
5. `05_winter_testing.ipynb` - Temporal validation

---

##  Methodology Highlights

### Data Cleaning Strategy

- **Short gaps (<2 hours)**: Linear interpolation
- **Medium gaps (2-24 hours)**: Forward fill
- **Long gaps (>24 hours)**: Flagged for special handling
- **Result**: 99.7% data retention with temporal integrity

### Feature Engineering Philosophy

Features designed to capture energy poverty indicators:

- **High variance**: Irregular consumption patterns
- **Weekend shifts**: More time at home for heating
- **Night consumption**: Electric heating (common in poverty)
- **Low load factors**: Curtailment behaviors

### Model Selection Rationale

**XGBoost chosen because:**
- Highest recall (87.8%) among tested models
- Built-in handling of missing values
- Provides feature importance metrics
- Computationally efficient at scale
- Strong interpretability with SHAP

---

##  Broader Impact & Applications

### Policy & Governance
- Automated screening for energy assistance programs
- Real-time monitoring of intervention effectiveness
- Evidence-based fuel poverty reduction strategies
- Support for net-zero transition equity

### Utility Providers
- Proactive identification of vulnerable customers
- Targeted energy efficiency programs
- Fair billing and payment plan development
- Customer support optimization

### Research Contributions
- Framework for privacy-preserving social analytics
- Benchmark for energy poverty detection methods
- Template for ethical ML in sensitive domains
- Demonstration of smart meter data potential

---

##  Ethical Framework

This research prioritizes:

✅ **Privacy**: No personally identifiable information required  
✅ **Transparency**: SHAP interpretability for explainable decisions  
✅ **Fairness**: Bias detection and mitigation strategies  
✅ **Recall Priority**: Minimize harm from missed vulnerable households  
✅ **Human Oversight**: ML as decision support, not replacement  

---

##  Academic Context

**MSc Dissertation**  
Applied Artificial Intelligence and Data Science  
Southampton Solent University  
December 2025

**Supervisor**: [Supervisor Name]

### Research Contribution

This work contributes to:
- Social impact AI and ML for public good
- Privacy-preserving analytics in sensitive domains
- Ethical considerations in algorithmic decision-making
- Smart meter data applications beyond energy management

---

##  Future Work

### Model Enhancements
- Ensemble methods combining multiple algorithms
- Deep learning approaches (LSTM, Transformers)
- Transfer learning across geographic regions
- Temporal modeling of poverty dynamics

### Feature Extensions
- Weather data integration
- Socioeconomic indicators from census data
- Energy tariff and pricing information
- Household composition estimates

### Deployment Considerations
- Real-time inference pipeline
- Model monitoring and drift detection
- Fairness auditing framework
- Integration with support systems

---

##  Acknowledgments

- **UK Power Networks** for the Low Carbon London dataset
- **Southampton Solent University** for academic supervision and support
- **Department of Business, Energy & Industrial Strategy** for context and motivation
- **Open-source community** for excellent ML libraries and tools

---

##  Author

**Papa Kwadwo Bona Owusu**   
MSc Applied AI & Data Science

---

## 📄 License

This project is part of academic research submitted for MSc degree requirements. 

For data usage, please refer to the [Low Carbon London dataset license](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households).

---

##  Contact & Collaboration

Interested in this research or potential collaboration? 

- Open an [Issue](https://github.com/Pehrry/energy-poverty-detection-ml/issues)
- Submit a [Pull Request](https://github.com/Pehrry/energy-poverty-detection-ml/pulls)
- Reach out via [email](mailto:your.email@example.com)

---

##  Repository Stats

- **Notebooks**: 5 comprehensive Jupyter notebooks
- **Features Engineered**: 90+ predictive features
- **Lines of Code**: ~3,500 (across all notebooks)
- **Data Processed**: 167 million observations
- **Model Performance**: 87.8% recall, 86.2% accuracy

---

##  Reproducibility Materials

The [`reproducibility/`](reproducibility/) directory contains a clean,
standalone companion to the exploratory notebooks above: a pinned
environment lock file, importable label-construction and
feature-generation code, annual and winter training/evaluation scripts
(with fixed random seeds and the bootstrap-CI procedure used throughout
the paper), the independently re-derived winter-only experiment, all
SHAP figure-generation scripts, model cards for the primary and
secondary models, and the underlying data sample itself. See
[`reproducibility/README.md`](reproducibility/README.md) for the full
structure and end-to-end run instructions.

---

##  Star This Repository

If you find this work useful for your research or interested in energy poverty detection, please consider starring the repository!

---

##  Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{owusu2025energy,
  title={Machine Learning for Energy Poverty Detection Using Smart Meter Consumption Patterns},
  author={Owusu, Papa Kwadwo Bona},
  year={2025},
  school={Southampton Solent University},
  type={MSc Dissertation},
  note={Applied Artificial Intelligence and Data Science}
}
```

---

** This research demonstrates how machine learning can support social policy through privacy-preserving, scalable, and ethical analytics in the fight against energy poverty.**

---

*Last updated: December 2025*
