# Employee Salary Prediction — Machine Learning Project
### TechWorks Consulting | Data Science PGC | Internshala Trainings

> **Author:** Chukwuemeka  
> **Project Type:** Supervised Machine Learning — Regression  
> **Notebook:** [📂 View Full Jupyter Notebook on Google Drive](https://drive.google.com/file/d/1Gx2WM6zKexUU-5IYgNVSImQcKgH9xzva/view?usp=sharing)

---

## 📋 Table of Contents

1. [Project Title](#-employee-salary-prediction--machine-learning-project)
2. [Project Overview / Problem Statement](#2-project-overview--problem-statement)
3. [Data Sources](#3-data-sources)
4. [Tools & Libraries](#4-tools--libraries)
5. [Data Cleaning & Preprocessing](#5-data-cleaning--preprocessing)
6. [Exploratory Data Analysis (EDA)](#6-exploratory-data-analysis-eda)
7. [Machine Learning Analysis](#7-machine-learning-analysis)
8. [Results](#8-results)
9. [Recommendations](#9-recommendations)
10. [Limitations](#10-limitations)
11. [Questions & Answers](#11-questions--answers)

---

## 2. Project Overview / Problem Statement

**TechWorks Consulting** is a leading IT staffing firm that places highly skilled technology professionals across businesses of all sizes. As the company scales rapidly and handles large-volume hiring projects, one of the key operational challenges it faces is **determining fair, consistent, and competitive salaries** for newly hired employees.

The company accounts for multiple factors when setting compensation:
- **Market rate** for a given skill set and job role
- **Employee qualifications** — including the tier of college attended and academic performance
- **Prior experience** — measured in months
- **Previous CTC** (Cost to Company) and job change history
- **Location** — whether the employee is based in a metro or non-metro city
- **Role** — whether the employee is a Manager or Executive
- **Employee performance** — factored in through performance evaluation systems

Manual approaches to salary setting are slow, inconsistent, and prone to bias. The goal of this project is to **build a machine learning regression model** that can predict an employee's salary (CTC) based on these factors — enabling TechWorks Consulting to make data-driven, fair, and efficient compensation decisions.

---

## 3. Data Sources

The project uses three CSV datasets merged and processed together:

| Dataset | Description |
|---|---|
| `ML case Study.csv` | Main employee dataset with 8 columns including college, city, role, previous CTC, experience, graduation marks, and target CTC |
| `Colleges.csv` | Reference list of colleges classified into Tier 1, Tier 2, and Tier 3 |
| `cities.csv` | Reference list of cities classified as Metro or Non-Metro |

### Dataset Columns (Main File)

| Column | Description |
|---|---|
| `College` | Name of college attended (encoded to Tier 1, 2, or 3) |
| `City` | Employee's city (Metro = 1, Non-Metro = 0) |
| `Role` | Job role — Manager or Executive (one-hot encoded) |
| `PreviousCTC` | Previous salary of the employee |
| `Previous_job_change` | Number of previous job changes |
| `Graduation_Marks` | Academic score at graduation |
| `EXP_(Month)` | Total experience in months |
| `CTC` | **Target variable** — current salary to be predicted |

---

## 4. Tools & Libraries

The following tools and Python libraries were used throughout this project:

| Category | Tool / Library |
|---|---|
| **Language** | Python 3 |
| **Data Manipulation** | `pandas`, `numpy` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Machine Learning** | `scikit-learn` |
| **Models Used** | `LinearRegression`, `Ridge`, `Lasso`, `RandomForestRegressor`, `GradientBoostingRegressor` |
| **Model Tuning** | `GridSearchCV` |
| **Pipeline & Preprocessing** | `Pipeline`, `ColumnTransformer`, `StandardScaler`, `RFE` |
| **Evaluation Metrics** | `mean_squared_error`, `r2_score` |
| **Notebook Environment** | Jupyter Notebook |

---

## 5. Data Cleaning & Preprocessing

Data preprocessing was a critical step performed before any model training. The steps below were followed:

### 5.1 Categorical Encoding — College Tier
College names were mapped to numerical tiers using a reference dataset:

```python
# Encoding logic
if college in Tier1:   → replaced with 1
elif college in Tier2: → replaced with 2
elif college in Tier3: → replaced with 3
```

**Rationale:** Tier 1 colleges carry the highest academic and market prestige and thus have a stronger influence on salary predictions.

### 5.2 Categorical Encoding — City Type
City names were mapped to binary numerical values using the cities reference file:

```python
# Encoding logic
if city in Metro_Cities:     → replaced with 1
elif city in Non_Metro_Cities: → replaced with 0
```

**Rationale:** Metro cities tend to offer higher salaries due to higher cost of living and more competitive job markets.

### 5.3 Dummy Variable Creation — Role
The `Role` column (Manager / Executive) was converted to dummy (binary) variables using `pd.get_dummies()` with `drop_first=True` to avoid multicollinearity:

```python
Chuks_employee_data = pd.get_dummies(Chuks_employee_data, columns=['Role'], drop_first=True)
# Resulting column: Role_Manager (1 = Manager, 0 = Executive)
```

### 5.4 Handling Missing Values
A check using `.info()` confirmed that **no null values** were present in the dataset, so no imputation was necessary.

### 5.5 Outlier Treatment
Outliers were detected through pairplot visualization and treated using a **winsorization approach** based on percentile capping. The 99th percentile was used as the upper bound for `PreviousCTC`:

```python
uv = np.percentile(Chuks_employee_data['PreviousCTC'], [99])[0]
Chuks_employee_data['PreviousCTC'] = Chuks_employee_data['PreviousCTC'].apply(
    lambda x: uv if x > 3 * uv else x
)
```

**Why this matters:** Extreme salary outliers, if left untreated, can skew regression models toward higher salary predictions and reduce overall accuracy.

### 5.6 Feature Scaling
`StandardScaler` was applied to all numerical features within a preprocessing pipeline to ensure no single feature dominates the model due to scale differences:

```python
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
```

### 5.7 Feature Selection
Recursive Feature Elimination (RFE) was used with a `LinearRegression` estimator to identify the top 5 most relevant features, though this was integrated within individual pipeline experiments.

---

## 6. Exploratory Data Analysis (EDA)

### 6.1 Pairplot Analysis
A pairplot was generated across all features to visually explore relationships between variables and identify patterns, correlations, and potential outliers.

```python
sns.pairplot(Chuks_employee_data)
```

**Key Observation:** Outliers were clearly visible in the `CTC` and `PreviousCTC` columns, particularly at the high end of the distribution, confirming the need for outlier treatment.

### 6.2 Correlation Matrix (Heatmap)
A heatmap of the correlation matrix was generated to understand which features have the strongest linear relationship with the target variable `CTC`:

```python
plt.figure(figsize=(12, 8))
corr_matrix = Chuks_employee_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
```

**Key Observations:**
- `PreviousCTC` and `EXP_(Month)` showed strong positive correlation with `CTC`
- `College` tier also demonstrated a meaningful association with salary level
- `Graduation_Marks` had a moderate positive relationship with CTC

### 6.3 Descriptive Statistics
`.describe()` was used to understand the distribution of each numeric feature, revealing ranges, mean, standard deviation, and quartile values across the dataset.

---

## 7. Machine Learning Analysis

### 7.1 Train-Test Split
The dataset was split into training (80%) and testing (20%) sets using a fixed random state for reproducibility:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 7.2 Models Evaluated
Five regression models were trained, tuned, and evaluated:

| Model | Description |
|---|---|
| **Linear Regression** | Baseline model; assumes a linear relationship between features and salary |
| **Ridge Regression** | Regularized linear model (L2 penalty); reduces overfitting |
| **Lasso Regression** | Regularized linear model (L1 penalty); also performs feature selection |
| **Random Forest Regressor** | Ensemble of decision trees; handles non-linearity and interactions |
| **Gradient Boosting Regressor** | Sequential ensemble; strong performance on structured tabular data |

### 7.3 Hyperparameter Tuning
`GridSearchCV` with 5-fold cross-validation was applied to all models. Key hyperparameter grids:

| Model | Hyperparameters Tuned |
|---|---|
| Ridge / Lasso | `alpha`: [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] |
| Random Forest | `n_estimators`: [50, 100, 200, 300]; `max_depth`: [None, 10, 20, 30]; `min_samples_split`: [2, 5, 10] |
| Gradient Boosting | `n_estimators`: [50, 100, 200, 300]; `learning_rate`: [0.01, 0.1, 0.2]; `max_depth`: [3, 5, 7] |

### 7.4 Evaluation Metrics
Each model was evaluated using:
- **MSE (Mean Squared Error):** Measures the average squared difference between predicted and actual CTC. Lower is better.
- **R² Score:** Measures the proportion of variance in CTC explained by the model. Higher is better (max = 1.0).

### 7.5 Model Comparison Visualization
A dual-axis chart was generated to compare all five models on MSE (bar chart) and R² Score (line chart) simultaneously:

```python
fig, ax1 = plt.subplots(figsize=(10, 6))
# Bar chart for MSE
bars_mse = ax1.bar(x, mse_values, width=bar_width, label='MSE', color='lightblue', alpha=0.7)
# Line chart for R²
ax2.plot(x, r2_values, label='R²', color='green', marker='o', linestyle='--', linewidth=2)
```

---

## 8. Results

After training and evaluating all five models with hyperparameter tuning using `GridSearchCV`, the results were as follows:

| Model | MSE | R² Score |
|---|---|---|
| Linear Regression | Higher | Lower |
| Ridge Regression | Moderate | Moderate |
| Lasso Regression | Moderate | Moderate |
| Gradient Boosting | Low | High |
| **Random Forest** | **Lowest** | **Highest** |

### Best Performing Model: Random Forest Regressor

The **Random Forest Regressor** achieved the **lowest MSE** and the **highest R² score** among all tested models.

**Why Random Forest performed best:**
- It is an **ensemble model** that combines multiple decision trees, reducing overfitting and increasing generalization
- It handles **non-linear relationships** in the data effectively — salary is rarely a simple linear function of experience or education
- It is inherently **robust to outliers**, which is particularly valuable given the salary distribution in this dataset
- It can naturally capture **feature interactions** (e.g., the combined effect of Tier 1 college + Metro city + high experience on salary)
- It performs **implicit feature selection** by averaging importance across trees

---

## 9. Recommendations

Based on the analysis and model results, the following recommendations are made for TechWorks Consulting:

1. **Deploy the Random Forest model** for initial salary estimation during the hiring process — it provides the most reliable predictions with the current dataset.

2. **Invest in data quality:** The model's accuracy is directly tied to data completeness. Ensure that candidate profiles include accurate values for previous CTC, experience duration, graduation marks, college name, and city before running predictions.

3. **Expand the feature set:** Consider incorporating additional relevant features such as:
   - Specific technical skills or certifications
   - Industry of previous employment
   - Number of years since graduation
   - Performance scores from assessments

4. **Regularly retrain the model:** Salary benchmarks evolve with market trends. Schedule periodic retraining of the model (e.g., quarterly or biannually) using fresh hiring data.

5. **Use the model as a decision-support tool**, not a strict determinant. HR professionals should review predictions alongside domain knowledge and current market surveys before finalizing offers.

6. **Explore XGBoost and LightGBM** as next-level ensemble alternatives that may further improve performance on this structured tabular data.

---

## 10. Limitations

The following limitations were identified during the project:

| Limitation | Explanation |
|---|---|
| **Limited dataset scope** | The model was trained on historical data from TechWorks Consulting. Predictions may not generalize well to roles or industries outside this dataset |
| **Static market representation** | Salary trends change over time. The model does not account for inflation, market shifts, or emerging skill premiums |
| **Binary city classification** | Cities are encoded as only Metro (1) or Non-Metro (0), which oversimplifies the cost-of-living differences across regions |
| **College tier encoding** | The three-level tier encoding is a simplification; institutions within the same tier can vary significantly in prestige and outcomes |
| **No temporal features** | The dataset does not contain hiring date or year, making it impossible to account for salary inflation over time |
| **Interpretability trade-off** | Random Forest, while accurate, is a black-box model — individual salary predictions cannot be easily explained to candidates or stakeholders |

---

## 11. Questions & Answers

The following questions were answered as part of the project deliverables within the Jupyter Notebook:

---

### Q1: Your views about the problem statement?

**Answer:**

TechWorks Consulting has defined a well-structured and practically significant problem: building a machine learning model to determine the salary of newly hired employees using structured historical data. The target variable (CTC) is a continuous numerical variable, which makes this a **regression problem** — well-suited for supervised machine learning.

The problem is realistic and impactful. Salary determination is a high-stakes process that affects both the company's competitiveness and employee satisfaction. By leveraging machine learning, TechWorks can move away from subjective or inconsistent manual methods toward a **data-driven, transparent, and scalable compensation framework**. The available features — experience, college tier, city type, role, previous CTC, and graduation marks — provide a solid foundation for building a predictive model. Overall, the problem statement is clear, achievable, and highly relevant to real-world HR operations.

---

### Q2: What will be your approach to solving this task?

**Answer:**

The approach followed to solve this task was structured in five systematic steps:

1. **Understanding the Dataset:** Thoroughly reviewed the main employee dataset along with the supporting college and cities reference files. Identified the nature of each feature and the target variable (CTC).

2. **Data Preparation:** Encoded college names using the tier classification (Tier 1 = 1, Tier 2 = 2, Tier 3 = 3); encoded city types as binary values (Metro = 1, Non-Metro = 0); applied `pd.get_dummies()` with `drop_first=True` to create dummy variables for the `Role` column. Handled missing values (none found), detected and treated outliers using percentile-based capping, and applied `StandardScaler` for feature normalization.

3. **Model Selection and Training:** Split the data into 80% training and 20% testing sets. Trained five regression models: Linear Regression, Ridge, Lasso, Random Forest, and Gradient Boosting.

4. **Hyperparameter Tuning:** Applied `GridSearchCV` with 5-fold cross-validation to optimize each model's hyperparameters and identify the best configuration.

5. **Model Evaluation and Documentation:** Evaluated all models using MSE and R² metrics, visualized the comparison, selected the best model, and documented all steps and findings within the Jupyter Notebook.

---

### Q3: What were the available ML model options you had to perform this task?

**Answer:**

For this regression task, the following five machine learning models were evaluated:

- **Linear Regression** — The simplest regression baseline; assumes a linear relationship between all input features and the target salary. Useful for establishing a performance benchmark.

- **Ridge Regression (L2 Regularization)** — A regularized extension of linear regression that penalizes large coefficients to reduce overfitting while retaining all features.

- **Lasso Regression (L1 Regularization)** — Similar to Ridge but applies an L1 penalty that can drive some coefficients to exactly zero, effectively performing feature selection.

- **Random Forest Regressor** — An ensemble of multiple decision trees trained on random subsets of the data and features. Excellent at capturing non-linear patterns and interactions between variables.

- **Gradient Boosting Regressor** — A sequential ensemble technique that builds trees one at a time, with each tree correcting the errors of the previous one. Often delivers high performance on structured data.

---

### Q4: Which model's performance is best and what could be the possible reason for that?

**Answer:**

After evaluating all five models with hyperparameter tuning, **Random Forest Regressor** delivered the best performance, achieving the **lowest Mean Squared Error (MSE)** and the **highest R² Score**.

**Possible reasons for Random Forest's superior performance:**

- **Ensemble strength:** By aggregating predictions from many individual decision trees, Random Forest reduces both bias and variance — leading to more stable and accurate predictions than any single model.

- **Non-linearity handling:** Salary is influenced by complex, non-linear combinations of features (e.g., a person from a Tier 1 college in a metro city with 60+ months of experience earns exponentially more than someone with only one of these attributes). Linear models cannot capture these interactions, while Random Forest does so naturally.

- **Robustness to outliers:** Random Forest is less sensitive to outliers in the training data because its ensemble averaging smooths out extreme predictions from individual trees.

- **Implicit feature selection:** Random Forest evaluates feature importance internally, giving more weight to the most predictive variables and reducing the influence of noise.

The visualization comparing MSE (bar) and R² (line) across all models confirmed this conclusion clearly.

---

### Q5: What steps can you take to improve this selected model's performance even further?

**Answer:**

The following strategies can further enhance the Random Forest model's performance:

1. **Advanced Hyperparameter Tuning:** Continue refining hyperparameters such as `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features` using more exhaustive `GridSearchCV` or `RandomizedSearchCV` with a broader parameter space.

2. **Explore Advanced Ensemble Algorithms:** Test **XGBoost**, **LightGBM**, or **CatBoost**, which are gradient-boosted tree frameworks known to outperform Random Forest on many tabular datasets through better regularization and more efficient training.

3. **Feature Engineering:** Create new derived features that may better capture salary drivers, such as:
   - Salary-to-experience ratio
   - Interaction terms (e.g., College Tier × City Type)
   - Categorical bucketing of experience ranges

4. **Expand the Dataset:** Collect more historical hiring records — more data typically improves model accuracy and generalization. Including more diverse roles and geographies would also improve coverage.

5. **Cross-Validation Strategy:** Use stratified or repeated k-fold cross-validation to get more robust performance estimates and reduce variance in model evaluation.

6. **Outlier Detection Refinement:** Apply more rigorous outlier detection (e.g., Isolation Forest or IQR-based methods) and evaluate their impact on model performance.

7. **Incorporate Additional Features:** Include variables like specific technical skills, years since graduation, industry sector of prior employment, or professional certifications.

---

## 🔗 Project Notebook

Access the full Jupyter Notebook with all code, visualizations, model outputs, and analysis here:

**[📂 Chukwuemeka_final_machine_learning_Project.ipynb — Google Drive](https://drive.google.com/file/d/1Gx2WM6zKexUU-5IYgNVSImQcKgH9xzva/view?usp=sharing)**

---

*This project was completed as part of the Data Science Post Graduate Certificate (PGC) program by Internshala Trainings.*
