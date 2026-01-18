# analysing-bd-and-ai

## Project Overview

This project contains a comprehensive series of **statistical analysis notebooks** focused on exploratory data analysis, correlation testing, linear regression modeling, and chi-squared testing using the Retail Customer Purchases Dataset (20,000 transactions).

**Student ID**: 24136959

---

## Project Structure

```
analysing-bd-and-ai/
├── data/
│   ├── raw/
│   │   └── data.csv                    # Primary dataset (20,000 retail transactions)
│   ├── raw1/                           # Backup or alternative raw data directory
│   └── processed/                      # Output directory for processed datasets
├── notebooks/
│   ├── 01-exploration.ipynb            # EDA using 4V Big Data framework
│   ├── 02-correlation.ipynb            # Correlation analysis and relationships
│   ├── 03-linearregression.ipynb       # Linear regression with assumptions testing
│   └── 04-chi-squared.ipynb            # Chi-squared testing for categorical data
├── outputs/                            # Directory for analysis outputs and visualisations
├── requirements.txt                    # Python package dependencies
└── README.md                           # This file
```

---

## Jupyter Notebooks Description

### **01-exploration.ipynb** — Exploratory Data Analysis (EDA) with 4V Framework
**Purpose**: Foundational exploration of the retail customer dataset using the **4V Big Data characteristics**.

**Key Analyses**:
- **Volume**: Dataset scale assessment (rows, columns, memory footprint), numeric feature distributions
- **Velocity**: Temporal patterns and data generation rates over time
- **Variety**: Data type diversity, categorical value distributions, structural heterogeneity
- **Veracity**: Data quality assessment (missing values, outliers via IQR method, correlation patterns)

**Learning Outcomes**:
- Understand dataset dimensionality and composition
- Identify temporal trends and seasonality
- Detect data quality issues (missing values, outliers)
- Establish baseline distributions for subsequent analyses

**Output**: Summary statistics, visualisations, and quality metrics

---

### **02-correlation.ipynb** — Correlation Analysis and Variable Relationships
**Purpose**: Investigate relationships between variables using correlation techniques.

**Key Analyses**:
- Pearson correlation for continuous variables
- Spearman rank correlation for ordinal relationships
- Correlation heatmaps and strength assessment
- Statistical significance testing of correlations

**Learning Outcomes**:
- Identify linear and monotonic relationships
- Understand correlation vs. causation
- Detect multicollinearity issues
- Select meaningful features for predictive modeling

**Output**: Correlation matrices, heatmaps, and relationship visualisations

---

### **03-linearregression.ipynb** — Linear Regression with Comprehensive Assumptions Testing
**Purpose**: Build predictive regression models with **rigorous validation of all four LINEAR regression assumptions**.

**Key Models**:
1. **Model 1**: Purchase Amount Prediction
2. **Model 2**: Satisfaction Score Prediction

**Four Assumptions Tested** (using acronym **LINEAR**):
1. **Linearity (L)**: Linear relationship between predictors and target
   - Method: Residual plots visual inspection
   - Expected: Random scatter around zero line

2. **Independence (I)**: Observations are independent (no autocorrelation)
   - Method: Durbin-Watson test
   - Criteria: DW ≈ 2.0 (1.5–2.5 acceptable)

3. **Homoscedasticity (H)**: Constant variance of residuals
   - Method: Breusch-Pagan test
   - Criteria: p-value > 0.05 indicates homoscedasticity

4. **Normality (N)**: Residuals follow normal distribution
   - Methods: Shapiro-Wilk, Anderson-Darling, Q-Q plots
   - Criteria: p-value > 0.05 indicates normality

**Learning Outcomes**:
- Build end-to-end regression pipelines with scikit-learn
- Validate statistical assumptions rigorously
- Interpret diagnostic plots (residual, Q-Q, scale-location)
- Understand remediation strategies for assumption violations
- Compare model performance and assumption compliance

**Output**: Model performance metrics, diagnostic visualisations, comparative analysis

---

### **04-chi-squared.ipynb** — Chi-Squared Testing for Categorical Data
**Purpose**: Test associations between categorical variables and evaluate goodness-of-fit.

**Key Analyses**:
- **Goodness-of-Fit Tests**: Evaluate if data matches expected distributions
- **Independence Tests**: Determine if categorical variables are related
- **Effect Size Measures**: Cramér's V and Phi coefficients
- **Hypothesis Testing**: Formulation of null and alternative hypotheses

**Statistical Formula**:
$$\chi^2 = \sum \frac{(\text{Observed} - \text{Expected})^2}{\text{Expected}}$$

**Key Assumptions**:
- Categorical/nominal data
- Independent observations
- Expected frequencies ≥ 5 (typically)

**Learning Outcomes**:
- Perform chi-squared independence tests
- Interpret contingency tables and cross-tabulations
- Measure strength of associations (effect sizes)
- Determine practical significance vs. statistical significance
- Make data-driven conclusions about categorical relationships

**Output**: Test statistics, p-values, effect size measures, contingency tables

---

## Runtime Requirements

### Python Version
- **Python 3.8+** (recommended 3.9 or 3.10)

### Required Packages
See [`requirements.txt`](requirements.txt) for complete dependency list.

**Core Libraries**:
- **Data Processing**: `pandas`, `numpy`
- **Visualisation**: `matplotlib`, `seaborn`
- **Statistical Testing**: `scipy`, `statsmodels`
- **Machine Learning**: `scikit-learn`
- **Jupyter**: `jupyter`, `jupyterlab`

### Installation

1. **Activate virtual environment** (macOS/Linux):
   ```bash
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Jupyter Lab**:
   ```bash
   jupyter lab
   ```

   Or use Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

---

## Data

### Dataset Location
- **Raw Data**: `data/raw/data.csv` (Retail customer data 20000 4539.csv renamed to data.csv)
- **Description**: Retail Customer Purchases Dataset with 20,000 transactions
- **Size**: ~20,000 rows × multiple columns including Age, Purchase Amount, Satisfaction Score, Product Category, Payment Method, etc.

**Important**: Always inspect the first few rows before processing:
```python
import pandas as pd
df = pd.read_csv('data/raw/data.csv')
print(df.head())
print(df.info())
```

### Output Locations
- **Processed Data**: `data/processed/`
- **Visualisations & Outputs**: `outputs/`

---

## Usage Workflow

### Sequential Analysis Path
1. **Start with 01-exploration.ipynb** → Understand dataset structure and quality
2. **Progress to 02-correlation.ipynb** → Identify variable relationships
3. **Advance to 03-linearregression.ipynb** → Build and validate predictive models
4. **Conclude with 04-chi-squared.ipynb** → Analyze categorical associations

### Running Individual Notebooks
Each notebook is **self-contained** and can be run independently:
```bash
jupyter lab notebooks/03-linearregression.ipynb
```

### Key Execution Steps
- Install requirements
- Ensure data file exists at `data/raw/data.csv`
- Run notebook cells sequentially
- Review visualisations and statistical outputs
- Check `outputs/` directory for saved figures

---

## Key Concepts Covered

| Concept | Notebook | Application |
|---------|----------|-------------|
| **4V Big Data Framework** | 01-exploration | Data characterisation |
| **Descriptive Statistics** | 01-exploration | Summary metrics |
| **Correlation Analysis** | 02-correlation | Variable relationships |
| **Linear Regression** | 03-linearregression | Predictive modeling |
| **Assumption Testing** | 03-linearregression | Model validity |
| **Hypothesis Testing** | 04-chi-squared | Statistical inference |
| **Effect Sizes** | 02-correlation, 04-chi-squared | Practical significance |

---

## Notes for Users

### Important Considerations
- **Large Sample Protection**: With n=20,000, the Central Limit Theorem provides robustness even if normality assumption is violated
- **Ordinal Data**: Model 2 (Satisfaction Score) is ordinal; consider ordinal regression as alternative
- **Missing Values**: Review data quality in 01-exploration before modeling
- **Relative Paths**: All notebooks use relative paths assuming execution from project root
- **Reproducibility**: Set random seeds for train-test splits to ensure reproducible results

### Common Issues & Solutions
| Issue | Solution |
|-------|----------|
| Data file not found | Check `data/raw/data.csv` exists; verify file path |
| Missing packages | Run `pip install -r requirements.txt` |
| Jupyter kernel error | Activate `.venv` before launching Jupyter |
| Low model R² | Review feature engineering; consider non-linear models |

---

## Learning Outcomes

Upon completing all analyses, students will understand:

✓ **Exploratory Data Analysis**: 4V framework for big data characterisation  
✓ **Statistical Relationships**: Correlation and association testing  
✓ **Predictive Modeling**: Building and validating regression models  
✓ **Assumption Validation**: Rigorous testing of regression assumptions  
✓ **Hypothesis Testing**: Formulating and testing statistical hypotheses  
✓ **Data Quality**: Identifying and handling data issues  
✓ **Interpretation**: Translating statistical results into actionable insights  

---

## Author & Metadata

**Student ID**: 24136959  
**Dataset**: Retail Customer Purchases (20,000 transactions)  
**Project Focus**: Statistical Analysis & Big Data Fundamentals  
**Created**: As part of MBA coursework in Analysis of Big Data and AI  

---

## License

Datasets and analyses are student work for coursework.
