# ⚡ Energy Poverty Detection Dashboard

Interactive Streamlit application for ML-based vulnerability identification using smart meter consumption data.

![Dashboard Preview](https://via.placeholder.com/800x400/1e3a8a/ffffff?text=Energy+Poverty+Detection+Dashboard)

## 🎯 Overview

This dashboard demonstrates a machine learning approach to identifying energy poverty risk from electricity consumption patterns. Built for the MSc dissertation "Machine Learning Approaches for Energy Poverty Detection Using Smart Meter Data."

### Key Features

- **📊 Real-time vulnerability monitoring** across household portfolio
- **🔍 Individual household analysis** with detailed risk assessment
- **📈 Model performance metrics** with confusion matrices and ROC curves
- **🧠 SHAP-based interpretability** explaining every prediction
- **❄️ Winter-specific analysis** validating seasonal performance
- **🎯 Evidence-based thresholds** for policy intervention

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd energy-poverty-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_streamlit.txt
```

### Running the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📊 Dashboard Pages

### 1. Dashboard Overview
- **Key Metrics**: Total households, vulnerable count, model recall, high-risk alerts
- **Visualizations**: Vulnerability distribution, ACORN breakdown
- **Alerts**: Priority cases requiring immediate attention

### 2. Household Analysis
- **Individual Assessment**: Detailed risk profile for selected household
- **Feature Comparison**: Radar chart vs average patterns
- **Action Recommendations**: Evidence-based intervention guidance

### 3. Model Performance
- **Model Comparison**: Recall, precision, F1, AUC across 4 models
- **Confusion Matrix**: XGBoost performance breakdown
- **ROC Curves**: Discrimination capability visualization
- **Selection Rationale**: Why XGBoost performs best

### 4. Feature Importance
- **Top 10 Features**: SHAP-based importance rankings
- **Category Breakdown**: Vulnerability indicators dominate
- **Feature Explanations**: Detailed descriptions with literature basis
- **Policy Thresholds**: Evidence-based cutpoints for intervention

### 5. Winter Analysis
- **Seasonal Validation**: Annual vs winter recall comparison
- **Winter Features**: Importance of cold-weather indicators
- **Consumption Patterns**: Vulnerable vs non-vulnerable winter behavior
- **Health Implications**: Link to excess winter deaths (Marmot Review)

### 6. About
- **Research Context**: Methodology and objectives
- **Key Findings**: 87.8% recall, self-disconnection importance
- **Contributions**: Methodological, empirical, practical
- **Limitations**: Honest assessment of constraints

## 🎯 Key Findings

### Model Performance
- **XGBoost Recall**: 87.8% (identifies 9 in 10 vulnerable households)
- **Winter Performance**: 89.1% (maintained/improved during critical period)
- **Top Feature**: self_disconnect_ratio (SHAP: 0.0234)

### Evidence-Based Thresholds
| Feature | Threshold | Interpretation |
|---------|-----------|----------------|
| self_disconnect_ratio | >0.10 | 10%+ active hours without electricity |
| winter_zero_ratio | >0.12 | 12%+ winter periods disconnected |
| evening_zero_ratio | >0.09 | 9%+ evenings with no consumption |
| consumption_volatility | >2.0 | Erratic day-to-day patterns |

### Feature Categories
1. **Vulnerability Indicators** (25 features) - 60% of importance
2. **Winter-Specific** (10 features) - 20% of importance
3. **Temporal Patterns** (20 features) - Supporting role
4. **Consumption Statistics** (17 features) - Context
5. **Load Profile** (15 features) - Supplementary
6. **Variability Metrics** (10 features) - Additional signals
7. **ACORN Demographics** (5 features) - Demographic context

## 📚 Data Requirements

### Input Data Format

The dashboard expects data in the following structure:

```python
# Household features (engineered_features.csv)
household_id, mean_consumption, self_disconnect_ratio, winter_zero_ratio, ...

# Model predictions (predictions.csv)
household_id, prediction_proba, predicted_vulnerable, actual_vulnerable

# Feature importance (shap_importance.csv)
feature, shap_value, category
```

### Sample Data

The current version includes **synthetic sample data** for demonstration. To use your own data:

1. Replace `load_sample_data()` function in `app.py`
2. Load your engineered features CSV
3. Load your model predictions CSV
4. Load your SHAP importance values

## 🔧 Configuration

### Customization Options

#### Vulnerability Threshold
Adjust in sidebar (default: 0.5)
```python
vulnerability_threshold = st.slider("Vulnerability Threshold", 0.0, 1.0, 0.5)
```

#### High Confidence Filter
Toggle in sidebar to show only high-confidence predictions (>80% or <20%)

#### Color Scheme
Modify CSS variables in `st.markdown()` custom styles:
```python
--primary-color: #1e3a8a;
--secondary-color: #3b82f6;
--accent-color: #ef4444;
```

## 📈 Deployment

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud
1. Push code to GitHub repository
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Select repository and branch
4. Deploy!

### Docker (Optional)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

```bash
docker build -t energy-poverty-dashboard .
docker run -p 8501:8501 energy-poverty-dashboard
```

## 🛠️ Technical Stack

- **Frontend**: Streamlit 1.29.0
- **Data Processing**: Pandas, NumPy
- **ML Framework**: XGBoost, LightGBM, Scikit-learn
- **Interpretability**: SHAP
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Python**: 3.10+

## 📊 System Architecture

```
app.py (main application)
├── Load Data (CSV files or sample generator)
├── Sidebar (navigation & settings)
└── Pages
    ├── Dashboard Overview
    ├── Household Analysis
    ├── Model Performance
    ├── Feature Importance
    ├── Winter Analysis
    └── About

Supporting Files:
├── requirements_streamlit.txt (dependencies)
├── README_STREAMLIT.md (documentation)
└── data/ (optional - your actual data files)
```

## 📚 Literature Basis

Key research informing the methodology:

- **Fell et al. (2020)**: Self-disconnection vulnerability signal
- **Hills (2012)**: Low Income High Costs (LIHC) indicator
- **Boardman (1991)**: Fuel poverty conceptualization
- **Marmot Review (2011)**: Health impacts of cold homes
- **Lundberg & Lee (2017)**: SHAP interpretability framework
- **Chen & Guestrin (2016)**: XGBoost algorithm

## ⚠️ Limitations

1. **Sample Data**: Current version uses synthetic data for demonstration
2. **No Real-Time Updates**: Static dataset, not live monitoring
3. **Simplified SHAP**: Full SHAP computation requires trained model
4. **UK-Specific**: Tailored to UK fuel poverty context

## 🔮 Future Enhancements

- [ ] Live data connection to smart meter APIs
- [ ] Real-time model retraining
- [ ] User authentication and access control
- [ ] Export reports (PDF, CSV)
- [ ] Email alerts for high-risk cases
- [ ] Multi-language support
- [ ] Mobile-responsive improvements
- [ ] Integration with CRM systems

## 👨‍💻 Developer

**Papa Kwadwo Bona Owusu**  
MSc Applied AI & Data Science  
Southampton Solent University  
2024-2025

**Supervisor:** Dr. Hamidreza Soltani

## 📄 License

This project is for research and educational purposes. Academic use permitted with appropriate citation.

## 🙏 Acknowledgments

- UK Power Networks for Low Carbon London dataset
- Dr. Hamidreza Soltani for supervision
- Southampton Solent University
- Streamlit for the excellent framework

## 📧 Contact

For questions or collaboration:
- GitHub: [Your GitHub Profile]
- LinkedIn: [Your LinkedIn]
- Email: [Your Email]

---

**Built with ❤️ for social impact through data science**
