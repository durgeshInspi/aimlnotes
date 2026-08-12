# Complete Machine Learning Notes for Students

---

## 📌 Master Table of Contents
1. [Module 1: Introduction to Machine Learning](#module-1-introduction-to-machine-learning)
   - What is Machine Learning?
   - Why Do We Use Machine Learning?
   - Types of Machine Learning (Supervised, Unsupervised, Semi-Supervised, Reinforcement) with Real Examples
2. [Module 2: Exploratory Data Analysis (EDA)](#module-2-exploratory-data-analysis-eda)
   - What is EDA & Why is it Important?
   - Step 1: Viewing the Data (`.head()`, `.tail()`, `.sample()`, `.info()`, `.shape`)
   - Step 2: Summary Statistics (`.describe()`, Central Tendency & Dispersion)
   - Step 3: Value Counts & Frequency Distribution (`.value_counts()`)
   - Step 4: Missing Values Identification & Visualization
   - Step 5: Data Visualization (Histograms, Boxplots, Bar Plots, Correlation, Heatmaps)
   - Step 6: Target Variable Exploration (Regression vs Classification)
3. [Module 3: Data Cleaning](#module-3-data-cleaning)
   - What is Data Cleaning?
   - Step 1: Handling Missing Values & Strategies (MCAR/MAR/MNAR, Drop, Mean/Median/Mode, KNN)
   - Step 2: Removing Duplicates (`.duplicated()`, `.drop_duplicates()`)
   - Step 3: Fixing Data Types (`.astype()`, `pd.to_numeric()`, `pd.to_datetime()`)
   - Step 4: Handling Inconsistent Categories & Typos (`.str.strip()`, `.str.lower()`, `.replace()`)
   - Step 5: Detecting & Handling Outliers (IQR Method, Z-Score Method, Capping, Trimming)
   - Step 6: Fixing Logic or Domain Errors (Invalid values, Logical date ordering, Range violations)
4. [Module 4: Data Preprocessing](#module-4-data-preprocessing)
   - What is Data Preprocessing?
   - Step 1: Encoding Categorical Variables (One-Hot, Ordinal, Label, Target Encoding)
   - Step 2: Feature Transformation (Log, Square Root, Box-Cox, Yeo-Johnson)
   - Step 3: Feature Scaling & Normalization (StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler)
   - Step 4: Train-Test Split & Preventing Data Leakage
5. [Module 5: Feature Engineering](#module-5-feature-engineering)
   - What is Feature Engineering?
   - Datetime Feature Extraction & Cyclical Sine/Cosine Encoding
   - Domain Ratios & Combinations
   - Text Feature Extraction (Length, Word Count, Character Count)
   - Feature Discretization / Binning (`pd.cut`, `pd.qcut`)
   - Group Aggregations (`groupby().transform()`)
6. [Module 6: Feature Selection](#module-6-feature-selection)
   - What is Feature Selection?
   - Filter Methods (Variance Threshold, Correlation Matrix & VIF, Chi-Square, ANOVA, Mutual Info)
   - Wrapper Methods (Forward Selection, Backward Elimination, RFE / RFECV)
   - Embedded Methods (Lasso L1 Regularization, Tree Importance, Permutation Importance)
   - Feature Selection Decision Matrix & Summary Checklist for Students

---

# Module 1: Introduction to Machine Learning

## 1.1 What is Machine Learning?

**Machine Learning (ML)** is a subfield of Artificial Intelligence (AI) where computers learn from **data** and **past experiences** to make predictions or decisions without being explicitly programmed with manual rules.

Traditional programming requires humans to write strict rules:
$$\text{Input Data} + \text{Explicit Rules} \longrightarrow \text{Output}$$

Machine Learning flips this paradigm:
$$\text{Input Data} + \text{Output Answers} \longrightarrow \text{Machine Learns Rules}$$

```
+-------------------------------------------------------------+
|                  TRADITIONAL PROGRAMMING                    |
|                                                             |
|   Data ------->  +------------------+                       |
|                  |  Rules / Logic   | -------> Output       |
|   Rules ------>  +------------------+                       |
+-------------------------------------------------------------+

+-------------------------------------------------------------+
|                     MACHINE LEARNING                        |
|                                                             |
|   Data ------->  +------------------+                       |
|                  | Machine Learning | -------> Learned Rules|
|   Output ----->  +------------------+          (Model)      |
+-------------------------------------------------------------+
```

---

### AI vs ML vs Deep Learning (DL)

- **Artificial Intelligence (AI)**: The broad concept of creating smart machines capable of performing tasks that typically require human intelligence.
- **Machine Learning (ML)**: A subset of AI that uses statistical techniques to enable machines to improve at tasks through experience/data.
- **Deep Learning (DL)**: A specialized subset of ML based on Multi-Layer Neural Networks that imitate the human brain.

---

## 1.2 Why Do We Use Machine Learning?

Before Machine Learning, software engineers wrote `if-else` rules for everything. But writing rules fails for complex real-world problems.

### Example: Email Spam Detection

**Without Machine Learning (Traditional If-Else)**:
```python
def is_spam(email_text):
    if "win money" in email_text or "free lottery" in email_text:
        return True
    return False
```
*Problem*: Spammers will quickly bypass this rule by writing `"w!n m0ney"` or `"free lotto"`. You would need millions of complex `if-else` statements, making maintenance impossible!

**With Machine Learning**:
You give 100,000 emails (labeled as `spam` or `not spam`) to an ML algorithm. The algorithm automatically learns patterns, word frequencies, combinations, and subtle clues. When spammers change techniques, you simply retrain the model with fresh data!

### Key Reasons to Use ML:
1. **Handles Complex & High-Dimensional Problems**: Problems like facial recognition, handwriting identification, or medical image diagnosis cannot be solved with manual rules.
2. **Adapts to Changing Data**: ML models dynamically update and adapt when fresh data arrives.
3. **Automation of Decision Making**: Processes millions of data points per second (e.g., credit card fraud detection in real-time).
4. **Discovers Hidden Insights**: Uncovers trends and associations in massive datasets that humans could never detect.

---

## 1.3 Types of Machine Learning

Machine Learning is broadly categorized into **4 major types**:

```
                       +-----------------------------+
                       |  Types of Machine Learning  |
                       +--------------+--------------+
                                      |
         +-----------------+----------+----------+-----------------+
         |                 |                     |                 |
+--------v-------+ +-------v--------+   +--------v-------+ +-------v--------+
|   Supervised   | |  Unsupervised  |   | Semi-Supervised| | Reinforcement  |
|    Learning    | |    Learning    |   |    Learning    | |    Learning    |
+----------------+ +----------------+   +----------------+ +----------------+
```

---

### Type 1: Supervised Learning

In **Supervised Learning**, the model learns under "supervision". The dataset contains both **Input Features ($X$)** and correct **Target Labels ($y$)**. The goal is to learn a mapping function from input to output: $y = f(X)$.

Supervised Learning is divided into **two categories**:

#### 1. Regression (Predicting Continuous Numbers)
The target output is a continuous numerical value (e.g., price, temperature, age, income).

- **Real-World Examples**:
  - **House Price Prediction**: Inputs: area (sq ft), bedrooms, location $\longrightarrow$ Target: Price ($\$350,000$).
  - **Stock Market Price Forecasting**: Inputs: past trend, volume, interest rate $\longrightarrow$ Target: Tomorrow's stock price.
  - **Salary Prediction**: Inputs: years of experience, education level $\longrightarrow$ Target: Salary ($\$85,000$).

#### 2. Classification (Predicting Categories / Labels)
The target output is a discrete category or class label.

- **Binary Classification** (2 classes):
  - **Email Spam Detection**: `Spam` vs `Not Spam`.
  - **Disease Detection**: `Cancerous` vs `Benign`.
  - **Bank Loan Approval**: `Approved` vs `Rejected`.
- **Multiclass Classification** (3+ classes):
  - **Handwritten Digit Recognition**: Predicting digits `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`.
  - **Customer Feedback Sentiment**: `Positive`, `Neutral`, `Negative`.

---

### Type 2: Unsupervised Learning

In **Unsupervised Learning**, the dataset contains only **Input Features ($X$)** without any target labels ($y$). The algorithm tries to discover hidden patterns, groupings, or structures in the data on its own.

Unsupervised Learning is divided into **three categories**:

#### 1. Clustering (Grouping Similar Data Points)
Automatically groups data points with similar characteristics together.

- **Real-World Examples**:
  - **Customer Segmentation**: E-commerce stores (Amazon, Flipkart) grouping customers by buying behavior into categories like *High Spenders*, *Bargain Hunters*, *Infrequent Buyers*.
  - **Document / News Clustering**: Grouping news articles automatically into *Sports*, *Politics*, *Tech*.

#### 2. Association Rule Learning (Finding Relationships)
Finds hidden relationships or dependencies between different items.

- **Real-World Example**:
  - **Market Basket Analysis (Retail Stores)**: "Customers who buy Bread and Butter are 80% likely to also buy Milk." Used by supermarkets to place related products on adjacent shelves.

#### 3. Dimensionality Reduction (Data Compression / Simplification)
Reduces the number of features (columns) while retaining essential information.

- **Real-World Example**: Reducing a 100-column dataset to 2 columns using **PCA (Principal Component Analysis)** for 2D visualization.

---

### Type 3: Semi-Supervised Learning

**Semi-Supervised Learning** lies between Supervised and Unsupervised Learning. The dataset contains a **small amount of labeled data** and a **large amount of unlabeled data**.

- **Why use it?** Labeling data manually (e.g., by medical doctors) is extremely expensive and time-consuming.
- **Real-World Examples**:
  - **Medical X-Ray Diagnosis**: A hospital has 500 labeled X-rays (verified by radiologists) and 50,000 unlabeled X-rays. The model trains on the labeled subset and pseudo-labels the rest to improve accuracy.
  - **Google Photos Tagging**: You tag 1 photo of your friend, and Google automatically identifies and groups all other photos of that person.

---

### Type 4: Reinforcement Learning (RL)

In **Reinforcement Learning**, an **Agent** learns how to behave in an **Environment** by performing **Actions** and receiving feedback in the form of **Rewards** (positive) or **Penalties** (negative). The agent learns through **trial and error** to maximize cumulative rewards.

```
                  +-------------------+
                  |    Environment    |
                  +---------+---------+
                            |
           State, Reward    |    Action
           (Feedback)       v    (Movement)
                  +-------------------+
                  |       Agent       |
                  +-------------------+
```

- **Core Components**:
  - **Agent**: The learner or decision maker.
  - **Environment**: The world the agent interacts with.
  - **Action ($A$)**: Move made by the agent.
  - **State ($S$)**: Current situation of the agent.
  - **Reward ($R$)**: Feedback score after taking an action.

- **Real-World Examples**:
  - **Self-Driving Cars (Tesla, Waymo)**: Learns driving by staying in lane (+reward) and avoiding collisions (-heavy penalty).
  - **Game AI (AlphaGo, Chess, OpenAI Five)**: Beating human grandmasters by playing millions of self-play games.
  - **Robotic Arm Automation**: Learning to pick and assemble items in factory warehouses.

---

## 💡 Summary Comparison Table of ML Types

| Feature | Supervised | Unsupervised | Semi-Supervised | Reinforcement |
| :--- | :--- | :--- | :--- | :--- |
| **Input Data** | Labeled Data ($X, y$) | Unlabeled Data ($X$) | Small labeled + Large unlabeled | Environment States |
| **Goal** | Predict Target ($y$) | Discover Hidden Patterns | Predict using mixed data | Maximize Rewards |
| **Feedback** | Direct Supervised | None | Partial Feedback | Rewards / Penalties |
| **Key Methods** | Regression, Classification | Clustering, Association | Self-Training | Q-Learning, PPO |
| **Example** | House Price, Spam Detection | Customer Segmentation | Medical Image Labeling | Self-Driving Cars |

---
---

# Module 2: Exploratory Data Analysis (EDA)

## 2.1 What is Exploratory Data Analysis (EDA)?

**Exploratory Data Analysis (EDA)** is the crucial first phase in data science and machine learning where we inspect, clean, summarize, and visualize a dataset to understand its underlying structure, distributions, anomalies, patterns, and relationships **before** building ML models.

> **Analogy**: EDA is like a doctor conducting diagnostic tests (blood tests, X-rays, stethoscope checks) on a patient before writing a prescription. Building a model without EDA is like prescribing medicine blindly!

---

## 2.2 The 6 Essential Steps of EDA (Step-by-Step Pipeline)

A complete professional EDA workflow follows **6 systematic steps**:

```
Step 1: Viewing the Data (.head, .tail, .sample, .shape, .info)
   │
Step 2: Summary Statistics (.describe, Mean, Median, Std, IQR)
   │
Step 3: Value Counts & Frequencies (.value_counts, .nunique)
   │
Step 4: Missing Values Identification & Pattern Analysis
   │
Step 5: Data Visualization (Histogram, Boxplot, Bar, Heatmap)
   │
Step 6: Target Variable Exploration (Regression vs Classification)
```

---

### Step 1: Viewing the Data

The very first action is to load the dataset and perform a quick structural inspection to answer:
- How many rows (records) and columns (features) exist?
- What are the column names and data types?
- What does the raw data actually look like?

```python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("housing_data.csv")

# 1. View first 5 rows
print("--- First 5 Rows ---")
print(df.head())

# 2. View last 5 rows
print("--- Last 5 Rows ---")
print(df.tail())

# 3. View 5 random sample rows (helps spot anomalies across rows)
print("--- Random 5 Sample Rows ---")
print(df.sample(5))

# 4. Check Dataset Dimensions (Rows, Columns)
print("Shape of Dataset (Rows, Cols):", df.shape)

# 5. Check Column Names, Non-Null Counts, and Data Types
print("--- Dataset Info ---")
print(df.info())
```

---

### Step 2: Summary Statistics

Summary statistics give a mathematical overview of the distribution, center, spread, and range of your features.

#### Numerical Summary (`df.describe()`)
Computes central tendency and dispersion metrics for continuous features:
- **Count**: Number of non-missing observations.
- **Mean ($\mu$)**: The arithmetic average.
- **Std ($\sigma$)**: Standard deviation (how spread out numbers are from the mean).
- **Min / Max**: Minimum and maximum observed values.
- **25% ($Q_1$)**: First quartile (25% of data falls below this value).
- **50% ($Q_2$ / Median)**: Median (middle value dividing data 50/50).
- **75% ($Q_3$)**: Third quartile (75% of data falls below this value).

```python
# Statistical summary for numerical columns
print("--- Numerical Summary ---")
print(df.describe())

# Statistical summary for categorical columns
print("--- Categorical Summary ---")
print(df.describe(include='object'))
```

> 💡 **Student Tip - Mean vs Median**:
> - If **Mean $\approx$ Median**, the feature follows a symmetric, bell-shaped (Normal) distribution.
> - If **Mean > Median**, the feature is **Right-Skewed** (has extreme large values/outliers).
> - If **Mean < Median**, the feature is **Left-Skewed** (has extreme small values/outliers).

---

### Step 3: Value Counts & Frequency Distribution

For categorical features (e.g., `City`, `Education_Level`, `Payment_Method`), we must check how many records fall into each unique category.

```python
# 1. Count unique values per categorical column
print("Unique count per column:")
print(df.nunique())

# 2. Detailed value counts for a single categorical variable
print("--- Category Frequencies ---")
print(df['Education_Level'].value_counts())

# 3. Percentage frequency distribution (Normalized)
print("--- Category Percentages (%) ---")
print(df['Education_Level'].value_counts(normalize=True) * 100)
```

---

### Step 4: Missing Values Identification

Missing values (`NaN`, `None`, `null`) cause errors in ML algorithms if left untreated. We must check which columns contain missing data and how severe the missingness is.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Calculate total missing values per column
missing_count = df.isnull().sum()

# 2. Calculate percentage of missing values per column
missing_percent = (df.isnull().sum() / len(df)) * 100

# Combine into a clean summary table
missing_df = pd.DataFrame({
    'Missing Count': missing_count,
    'Percentage (%)': missing_percent
})

# Display columns that have missing values
print("--- Columns with Missing Values ---")
print(missing_df[missing_df['Missing Count'] > 0])

# 3. Visualize Missing Data using Seaborn Heatmap
plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title("Missing Values Pattern Heatmap (Yellow = Missing)")
plt.xlabel("Columns")
plt.show()
```

---

### Step 5: Data Visualization

Visualizations make data distributions, relationships, and anomalies instantly apparent to humans.

#### A. Histogram (Analyzing Single Numerical Distribution & Skewness)
Histograms group continuous numbers into intervals (bins) to show distribution shape.

```python
plt.figure(figsize=(8, 4))
sns.histplot(df['Income'], kde=True, bins=30, color='royalblue')
plt.title("Income Distribution (Histogram + KDE Curve)")
plt.xlabel("Income ($)")
plt.ylabel("Frequency")
plt.show()
```

#### B. Boxplot (Detecting Spread & Outliers)
Boxplots visually display the 5-number summary ($Min, Q_1, Median, Q_3, Max$) and mark outliers as individual points outside the whiskers.

```
       Outlier
          *
  |-------+-----------+-------|
 Min     Q1  Median  Q3      Max
```

```python
plt.figure(figsize=(6, 4))
sns.boxplot(y=df['Salary'], color='coral')
plt.title("Salary Boxplot (Identify Outliers & Spread)")
plt.ylabel("Salary")
plt.show()
```

#### C. Bar Plot (Comparing Categorical Frequencies)
Bar charts display discrete categories alongside their counts or mean target values.

```python
plt.figure(figsize=(8, 4))
sns.countplot(x='Department', data=df, palette='Set2')
plt.title("Employee Count by Department")
plt.xlabel("Department")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()
```

#### D. Correlation Matrix & Heatmap (Bivariate / Multivariate Analysis)
Correlation measures the strength and direction of linear relationship between two numerical variables. Pearson correlation coefficient ($r$) ranges from **-1.0 to +1.0**:
- **+1.0**: Perfect Positive Linear Correlation (As $X$ increases, $Y$ increases).
- **0.0**: No Linear Relationship.
- **-1.0**: Perfect Negative Linear Correlation (As $X$ increases, $Y$ decreases).

```python
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr(numeric_only=True)

sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix Heatmap")
plt.show()
```

---

### Step 6: Target Variable Exploration

The **Target Variable ($y$)** is the column your ML model tries to predict. You must analyze it differently based on your problem type:

#### A. Regression Target (Continuous Number e.g., `House_Price`)
- Inspect target distribution for skewness using Histogram & KDE.
- Check if target requires log transformation for linear models.

```python
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.histplot(df['House_Price'], kde=True, color='purple')
plt.title("Raw Target Distribution (Right-Skewed)")

plt.subplot(1, 2, 2)
sns.histplot(np.log1p(df['House_Price']), kde=True, color='green')
plt.title("Log-Transformed Target (Normal Curve)")

plt.tight_layout()
plt.show()
```

#### B. Classification Target (Categorical Label e.g., `Churn` or `Fraud`)
- Check for **Class Imbalance** (e.g., 95% Non-Fraud vs 5% Fraud).
- Highly imbalanced targets require specialized techniques like SMOTE or class weighting!

```python
plt.figure(figsize=(5, 4))
sns.countplot(x='Is_Fraud', data=df, palette='dark')
plt.title("Target Class Balance Check")
plt.show()

# Calculate proportions
print(df['Is_Fraud'].value_counts(normalize=True) * 100)
```

---
---

# Module 3: Data Cleaning

## 3.1 What is Data Cleaning?

**Data Cleaning** is the systematic process of fixing, correcting, or removing incomplete, duplicate, incorrectly formatted, corrupted, or outlier data within a raw dataset.

> **Crucial Difference for Students**:
> - **Data Cleaning**: Fixes **errors, missingness, and inconsistencies** in raw data (e.g., dropping duplicates, fixing typos, handling nulls).
> - **Data Preprocessing**: Transforms **valid clean data** into numerical/scaled formats ready for ML algorithms (e.g., scaling ranges, one-hot encoding).

---

## 3.2 The 6 Core Data Cleaning Steps

```
Step 1: Handling Missing Values & Strategies
   │
Step 2: Removing Duplicates
   │
Step 3: Fixing Data Types
   │
Step 4: Handling Inconsistent Categories & Typos
   │
Step 5: Detecting and Handling Outliers
   │
Step 6: Fixing Logic or Domain Errors
```

---

### Step 1: Handling Missing Values & Strategies

Missing data occurs when no value is stored for a feature in an observation.

#### Types of Missing Data Mechanics:
1. **MCAR (Missing Completely at Random)**: Missingness is purely random (e.g., survey page got lost in mail).
2. **MAR (Missing at Random)**: Missingness depends on observed data (e.g., younger people skip answering income).
3. **MNAR (Missing Not at Random)**: Missingness depends on the missing value itself (e.g., people with extremely high debt intentionally omit debt amount).

#### Strategies to Handle Missing Data:

```
                           +------------------------+
                           | Handling Missing Data  |
                           +-----------+------------+
                                       |
                 +---------------------+--------------------+
                 |                                         |
        +--------v-------+                        +--------v-------+
        |  Deletion      |                        |  Imputation    |
        |  (Drop Rows)   |                        |  (Fill Values) |
        +----------------+                        +----------------+
```

##### Strategy A: Deletion (`dropna`)
- **Row Deletion**: Drop rows with missing values if missing rows are <5% of total dataset.
- **Column Deletion**: Drop an entire column if >50% of its values are missing.

```python
# Drop rows where target column 'Exam_Score' is missing
df_clean = df.dropna(subset=['Exam_Score'])

# Drop column if >50% values are missing
threshold = len(df) * 0.5
df_clean = df.dropna(thresh=threshold, axis=1)
```

##### Strategy B: Simple Statistical Imputation
- **Mean Imputation**: Fill missing numerical values with column **Mean**. *Best for Normally Distributed data without outliers*.
- **Median Imputation**: Fill missing numerical values with column **Median**. *Best for Skewed data or features with Outliers*.
- **Mode Imputation**: Fill missing categorical values with the **Most Frequent Class (Mode)**.

```python
# 1. Median Imputation for Numerical Column
median_income = df['Income'].median()
df['Income'].fillna(median_income, inplace=True)

# 2. Mode Imputation for Categorical Column
mode_city = df['City'].mode()[0]
df['City'].fillna(mode_city, inplace=True)
```

##### Strategy C: Advanced Machine Learning Imputation (`KNNImputer`)
Uses $k$-Nearest Neighbors algorithm to predict missing values based on similarity to neighboring rows.

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df[['Age', 'Income']] = imputer.fit_transform(df[['Age', 'Income']])
```

---

### Step 2: Removing Duplicates

Duplicate rows waste memory, skew statistical distributions, and cause model overfitting.

```python
# 1. Check total number of duplicate rows
print("Duplicate rows count:", df.duplicated().sum())

# 2. View actual duplicate rows
print(df[df.duplicated()])

# 3. Drop duplicate rows (Keep first occurrence)
df.drop_duplicates(keep='first', inplace=True)
print("Shape after removing duplicates:", df.shape)
```

---

### Step 3: Fixing Data Types

Raw datasets often import numbers as strings or dates as generic objects. Machine Learning models require appropriate data types.

```python
# 1. Convert Object/String to Numeric (Coerce errors to NaN if invalid text exists)
df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')

# 2. Convert String to Datetime
df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], format='%Y-%m-%d')

# 3. Cast Float to Integer
df['Age'] = df['Age'].astype('int64')

# 4. Convert High-Cardinality String to Category (Saves Memory)
df['State'] = df['State'].astype('category')
```

---

### Step 4: Handling Inconsistent Categories & Typos

Human data entry leads to inconsistent text variations (e.g., `'Male'`, `'male'`, `'M'`, `' Male  '`).

```python
# 1. Strip leading and trailing whitespace
df['Gender'] = df['Gender'].astype(str).str.strip()

# 2. Convert all text to lower/title case for uniformity
df['Gender'] = df['Gender'].str.lower()

# 3. Replace inconsistent spelling variations with unified labels
gender_mapping = {
    'm': 'male',
    'male': 'male',
    'f': 'female',
    'femal': 'female',
    'female': 'female'
}
df['Gender'] = df['Gender'].map(gender_mapping)

print("Cleaned Category Distribution:")
print(df['Gender'].value_counts())
```

---

### Step 5: Detecting and Handling Outliers

Outliers are extreme data points that deviate significantly from the rest of the observations.

#### Detection Method 1: Interquartile Range (IQR Method)
$$\text{IQR} = Q_3 - Q_1$$
$$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}$$
$$\text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$

```python
# Calculate Q1, Q3, and IQR
Q1 = df['Income'].quantile(0.25)
Q3 = df['Income'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Income'] < lower_bound) | (df['Income'] > upper_bound)]
print(f"Total IQR Outliers Detected: {len(outliers)}")
```

#### Detection Method 2: Z-Score Method (For Normally Distributed Data)
$$Z = \frac{X - \mu}{\sigma}$$
Any data point with $|Z| > 3$ is considered an outlier (beyond $99.7\%$ of Gaussian distribution).

```python
from scipy import stats

z_scores = np.abs(stats.zscore(df['Height']))
outliers_z = df[z_scores > 3]
print(f"Z-Score Outliers Detected: {len(outliers_z)}")
```

#### Strategies to Handle Outliers:

1. **Trimming / Dropping**: Remove outlier rows if they are data errors.
2. **Capping / Winsorization**: Cap extreme values at lower/upper thresholds so data points are retained without distorting the model.

```python
# Capping using IQR boundaries
df['Income_Capped'] = np.where(df['Income'] > upper_bound, upper_bound,
                       np.where(df['Income'] < lower_bound, lower_bound, df['Income']))
```

---

### Step 6: Fixing Logic or Domain Errors

Domain errors occur when values are mathematically or logically impossible in the real world.

- **Examples**:
  - `Age = -5` or `Height = 0`
  - `Test_Score = 150%` (when max is 100)
  - `End_Date < Start_Date`

```python
# 1. Fix impossible negative values (Replace with NaN or cap at min valid value)
df.loc[df['Age'] <= 0, 'Age'] = np.nan

# 2. Fix range boundary violations (Score capped between 0 and 100)
df['Score'] = df['Score'].clip(lower=0, upper=100)

# 3. Fix Logical Date Ordering
invalid_dates = df[df['End_Date'] < df['Start_Date']]
print(f"Found {len(invalid_dates)} rows where End Date is before Start Date!")

# Drop invalid logical rows
df = df[df['End_Date'] >= df['Start_Date']]
```

---
---

# Module 4: Data Preprocessing

## 4.1 What is Data Preprocessing?

**Data Preprocessing** transforms clean, error-free raw data into an optimized, mathematically structured numerical representation suitable for training Machine Learning models.

---

## 4.2 Core Data Preprocessing Steps

```
Step 1: Encoding Categorical Variables (One-Hot, Ordinal, Label, Target)
   │
Step 2: Feature Transformation (Log, Square Root, Box-Cox, Yeo-Johnson)
   │
Step 3: Feature Scaling & Normalization (StandardScaler, MinMaxScaler, RobustScaler)
   │
Step 4: Train-Test Split & Preventing Data Leakage
```

---

### Step 1: Encoding Categorical Variables

Machine Learning models compute algebraic matrices ($Y = XW + b$). They cannot process raw text strings directly.

#### A. One-Hot Encoding (For Nominal Features with No Natural Rank)
Creates a new binary column (`0` or `1`) for every unique category.

```
Original: Color         One-Hot Encoded:
| Color |            | Color_Red | Color_Blue | Color_Green |
| Red   |    --->    |     1     |     0      |      0      |
| Blue  |            |     0     |     1      |      0      |
| Green |            |     0     |     0      |      1      |
```

> ⚠️ **The Dummy Variable Trap**:
> Including all $k$ dummy columns introduces perfect multicollinearity ($Color\_Green = 1 - Color\_Red - Color\_Blue$). Always use `drop='first'` to keep $k-1$ columns for linear models!

```python
# Option 1: Pandas get_dummies
df_encoded = pd.get_dummies(df, columns=['Color'], drop_first=True)

# Option 2: Scikit-Learn OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(drop='first', sparse_output=False)
encoded_array = ohe.fit_transform(df[['Color']])
```

#### B. Ordinal Encoding (For Ordinal Features with Natural Ranking)
Assigns ordered integer numbers ($0, 1, 2, 3$) reflecting natural category rank (e.g., `High School` < `Bachelors` < `Masters` < `PhD`).

```python
from sklearn.preprocessing import OrdinalEncoder

education_order = [['High School', 'Bachelors', 'Masters', 'PhD']]
ordinal_enc = OrdinalEncoder(categories=education_order)

df['Education_Encoded'] = ordinal_enc.fit_transform(df[['Education_Level']])
```

#### C. Label Encoding (For Target Labels $y$ ONLY)
Converts target classification labels into integers ($0, 1, 2$).

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
```

---

### Step 2: Feature Transformation

Feature transformation changes the mathematical shape of feature distributions to align with linear model assumptions (Gaussian Normal Distribution).

```
   Right-Skewed Distribution           Transformed Gaussian Curve
       |*                                      |   *   
       |* *                                  | * * * 
       |* * * *                            |* * * * *
       +------------------->               +------------------->
```

#### 1. Log Transformation ($\log(x + 1)$)
Compresses extreme right-skewed numerical tails (e.g., `Income`, `House_Price`, `Revenue`).

```python
df['Income_Log'] = np.log1p(df['Income'])  # log1p computes log(1 + x) to handle 0 values safely
```

#### 2. Square Root Transformation ($\sqrt{x}$)
Weaker transformation than Log; used for moderately right-skewed count data.

```python
df['Count_Sqrt'] = np.sqrt(df['Transaction_Count'])
```

#### 3. Power Transformations (Box-Cox & Yeo-Johnson)
Automatically finds the optimal power parameter $\lambda$ to transform skewed features into bell curves.
- **Box-Cox**: Requires strictly strictly positive values ($x > 0$).
- **Yeo-Johnson**: Works on both positive and negative values ($x \in \mathbb{R}$).

```python
from sklearn.preprocessing import PowerTransformer

# Yeo-Johnson Transformer
pt = PowerTransformer(method='yeo-johnson')
df['Income_Transformed'] = pt.fit_transform(df[['Income']])
```

---

### Step 3: Feature Scaling & Normalization

Distance-based algorithms (KNN, K-Means, SVM) and Gradient Descent optimization models (Linear/Logistic Regression, Neural Networks) are sensitive to feature scales. If `Salary` ranges from $10,000$ to $500,000$ and `Age` ranges from $18$ to $65$, `Salary` will dominate distance formulas!

```
Without Scaling: Distance = sqrt((Salary2 - Salary1)^2 + (Age2 - Age1)^2)
                 Salary differences drown out Age completely!
```

---

#### Comparison of Scaling Methods:

#### 1. Standardization (`StandardScaler`)
Scales feature to have a **Mean ($\mu$) of 0** and **Standard Deviation ($\sigma$) of 1**.
$$Z = \frac{X - \mu}{\sigma}$$

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['Age', 'Salary']])
```

#### 2. Min-Max Normalization (`MinMaxScaler`)
Rescales features strictly into a bounded range, typically **[0, 1]**.
$$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

```python
from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X[['Age', 'Salary']])
```

#### 3. Robust Scaling (`RobustScaler`)
Scales using **Median ($Q_2$)** and **Interquartile Range ($\text{IQR} = Q_3 - Q_1$)**.
$$\text{Scaled} = \frac{X - Q_2}{Q_3 - Q_1}$$
*Best for features containing unavoidable outliers!*

```python
from sklearn.preprocessing import RobustScaler

robust = RobustScaler()
X_robust = robust.fit_transform(X[['Age', 'Salary']])
```

---

#### 💡 Scaler Decision Guide Table for Students:

| Algorithm Type | Requires Feature Scaling? | Preferred Scaler |
| :--- | :--- | :--- |
| **Linear / Logistic Regression, SVM** | ✅ **YES** | `StandardScaler` |
| **K-Nearest Neighbors (KNN), K-Means** | ✅ **YES** | `StandardScaler` or `MinMaxScaler` |
| **Neural Networks, Image Pixel Processing** | ✅ **YES** | `MinMaxScaler` (Bounded [0, 1]) |
| **Datasets with Extreme Outliers** | ✅ **YES** | `RobustScaler` |
| **Decision Trees, Random Forest, XGBoost** | ❌ **NO** | Scale Invariant (Tree splits don't care!) |

---

### Step 4: Train-Test Split & Preventing Data Leakage

To evaluate model performance realistically, we split data into **Training Set (80%)** and **Testing Set (20%)**.

```
+-------------------------------------------------------+
|                 FULL DATASET (100%)                   |
+------------------------------------+------------------+
|        Train Set (80%)             |  Test Set (20%)  |
|  (Used to fit ML model)            | (Used for eval)  |
+------------------------------------+------------------+
```

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['Target'])
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y if y.nunique() < 10 else None
)
```

> ⚠️ **CRITICAL RULE - Preventing Data Leakage**:
> **Data Leakage** occurs when information from the test dataset leaks into the training pipeline.
> - **WRONG**: `scaler.fit_transform(X)` on full dataset before splitting. (Leaks test mean/std!)
> - **RIGHT**: `scaler.fit_transform(X_train)` on Train set, then `scaler.transform(X_test)` on Test set!

```python
# CORRECT WAY TO SCALE WITHOUT LEAKAGE
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)  # FIT and TRANSFORM on Train ONLY
X_test_scaled = scaler.transform(X_test)        # TRANSFORM ONLY on Test!
```

---
---

# Module 5: Feature Engineering

## 5.1 What is Feature Engineering?

**Feature Engineering** is the art and science of leveraging domain knowledge to transform raw variables into **new, more informative input features** that enhance the predictive power of Machine Learning models.

> *"Coming up with features is complicated, time-consuming, requires domain knowledge. Applied machine learning is basically feature engineering."* — Prof. Andrew Ng

---

## 5.2 Key Feature Engineering Techniques

### 1. Datetime Feature Extraction & Cyclical Encoding

A raw timestamp string like `"2026-08-12 17:35:00"` cannot be processed directly by algorithms. We extract granular components:

```python
df['Date'] = pd.to_datetime(df['Timestamp'])

# Standard DateTime Components
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
df['Hour'] = df['Date'].dt.hour
```

#### Cyclical Sine/Cosine Encoding (For Hours/Months)
Hours wrap around from $23 \longrightarrow 0$. Numerical values $23$ and $0$ seem far apart to ML, but they are only 1 hour apart! We transform cyclical time into 2D Sine/Cosine coordinates:
$$x_{sin} = \sin\left(\frac{2\pi \cdot t}{T}\right), \quad x_{cos} = \cos\left(\frac{2\pi \cdot t}{T}\right)$$

```python
# Cyclical Hour Encoding (T = 24)
df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
```

---

### 2. Domain Ratios & Combinations

Combining two or more columns into a meaningful business metric:

```python
# 1. Total Income in Loan Prediction
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']

# 2. Debt-to-Income Ratio
df['Debt_Income_Ratio'] = df['Total_Debt'] / (df['Total_Income'] + 1)

# 3. Price per Square Foot in Real Estate
df['Price_Per_SqFt'] = df['House_Price'] / df['Total_SqFt']

# 4. Body Mass Index (BMI) in Healthcare
df['BMI'] = df['Weight_kg'] / ((df['Height_cm'] / 100) ** 2)
```

---

### 3. Text Feature Extraction

Extracting numerical metadata from raw text columns (e.g., customer reviews, product descriptions):

```python
# 1. Character Length of Text
df['Review_Char_Count'] = df['Review_Text'].astype(str).apply(len)

# 2. Word Count
df['Review_Word_Count'] = df['Review_Text'].astype(str).apply(lambda x: len(x.split()))

# 3. Presence of Special Characters (e.g., Exclamation marks indicating emotion)
df['Exclamation_Count'] = df['Review_Text'].astype(str).apply(lambda x: x.count('!'))
```

---

### 4. Feature Discretization / Binning

Binning converts continuous numerical values into categorical intervals/ranges.

- **Fixed-Width Binning (`pd.cut`)**: Equal distance intervals.
- **Quantile Binning (`pd.qcut`)**: Equal sample frequency intervals per bin.

```python
# Binning Age into Discrete Life Stages
bins = [0, 12, 19, 35, 60, 100]
labels = ['Child', 'Teens', 'Young Adult', 'Middle Aged', 'Senior']

df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
```

---

### 5. Group Aggregations (`groupby().transform()`)

Creating statistical summary features relative to a customer's demographic or category group:

```python
# 1. Average customer spending by City
df['City_Avg_Spend'] = df.groupby('City')['Total_Spend'].transform('mean')

# 2. Difference between customer spend and their city's average spend
df['Spend_Diff_From_City_Avg'] = df['Total_Spend'] - df['City_Avg_Spend']
```

---
---

# Module 6: Feature Selection

## 6.1 What is Feature Selection?

**Feature Selection** is the process of selecting a subset of the most relevant and informative features to use in model building, while dropping noisy, uninformative, or redundant columns.

> **Feature Selection vs Dimensionality Reduction (PCA)**:
> - **Feature Selection**: Keeps a subset of the **original features intact** (preserves interpretability).
> - **Dimensionality Reduction**: Combines features into **brand new synthesized components** (loses feature names).

---

## 6.2 Primary Feature Selection Methods

```
                      +-----------------------------+
                      |  Feature Selection Methods  |
                      +--------------+--------------+
                                     |
         +---------------------------+---------------------------+
         |                           |                           |
+--------v-------+          +--------v-------+          +--------v-------+
| Filter Methods |          | Wrapper Methods|          | Embedded Method|
| (Statistical)  |          | (Iterative ML) |          | (In-Algorithm) |
+----------------+          +----------------+          +----------------+
```

---

### Method 1: Filter Methods (Fast & Model-Independent)

Filter methods evaluate the statistical connection between each feature and target variable **without training ML models**.

#### A. Variance Thresholding
Removes constant or near-constant features (features with variance lower than a threshold).

```python
from sklearn.feature_selection import VarianceThreshold

# Remove columns where >99% of rows have the exact same value
selector = VarianceThreshold(threshold=0.01)
X_high_var = selector.fit_transform(X)
```

#### B. Correlation Matrix & Multicollinearity (VIF)
If two input features are >85% correlated, drop one to avoid redundancy.

```python
corr_matrix = df.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Identify features with correlation higher than 0.85
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]
df_selected = df.drop(columns=to_drop)
```

#### C. Statistical Scoring Tests
- **Chi-Square ($\chi^2$)**: For Categorical Feature vs Categorical Target.
- **ANOVA F-Test**: For Numerical Feature vs Categorical Target.
- **Mutual Information**: Measures non-linear dependency between features and target.

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Select top 5 numerical features using ANOVA F-test
selector = SelectKBest(score_func=f_classif, k=5)
X_top5 = selector.fit_transform(X_train, y_train)
```

---

### Method 2: Wrapper Methods (Iterative Model Evaluation)

Wrapper methods evaluate combinations of features by training ML models iteratively and assessing performance scores.

#### A. Forward Selection
Starts with 0 features and adds the best performing feature one by one.

#### B. Backward Elimination
Starts with all features and iteratively removes the weakest feature.

#### C. Recursive Feature Elimination (RFE)
Fits an estimator and recursively prunes features with the lowest weights/importance.

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
rfe = RFE(estimator=model, n_features_to_select=5)
rfe.fit(X_train, y_train)

# View mask of selected features
selected_features = X.columns[rfe.support_]
print("RFE Selected Features:", selected_features)
```

---

### Method 3: Embedded Methods (Built-in Algorithm Selection)

Embedded methods perform feature selection automatically during model training.

#### A. Lasso Regression (L1 Regularization)
Lasso adds an L1 penalty term ($\lambda \sum |\beta_j|$) to the loss function that shrinks uninformative feature coefficients to **exact zero**, effectively dropping them!

```python
from sklearn.linear_model import LassoCV

lasso = LassoCV(cv=5).fit(X_train, y_train)
lasso_features = X.columns[lasso.coef_ != 0]
print("Lasso Selected Features:", lasso_features)
```

#### B. Tree-Based Feature Importances
Tree models calculate feature importance based on Gini impurity reduction across all splits.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier().fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X.columns)

# Plot Top 10 Features
plt.figure(figsize=(8, 5))
importances.nlargest(10).plot(kind='barh', color='teal')
plt.title("Random Forest Top 10 Feature Importances")
plt.xlabel("Gini Importance Score")
plt.show()
```

---

## 📌 Feature Selection Decision Matrix

| Method | Speed | Model-Dependent? | Captures Feature Interactions? | Best Recommended Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Filter Methods** | ⚡ Very Fast | ❌ No | ❌ No | Initial quick cleanup of massive datasets |
| **Wrapper Methods (RFE)** | 🐢 Slow | ✅ Yes | ✅ Yes | Small datasets requiring peak accuracy |
| **Embedded (Lasso / Trees)** | 🚀 Fast | ✅ Yes | ✅ Yes | Industry Standard for production pipelines |

---

## 🎓 Summary Checklist for Students

When working on any Machine Learning assignment or project, follow this exact workflow:

1. 🔍 **Problem Understanding**: Determine whether the problem is Supervised (Regression vs Classification) or Unsupervised.
2. 📊 **Exploratory Data Analysis (EDA)**: Inspect shape, inspect head/tail, calculate describe statistics, check value counts, analyze missingness heatmaps, plot distributions (histograms, boxplots), check correlations, and analyze target balance.
3. 🧹 **Data Cleaning**: Handle missing data (Impute/Drop), remove duplicate rows, convert data types, fix inconsistent category string spellings/typos, detect/cap outliers, and correct logical domain errors.
4. ⚙️ **Data Preprocessing**: Encode categorical variables (One-Hot/Ordinal), transform skewed distributions, apply feature scaling (`StandardScaler`/`MinMaxScaler`), and split into Train/Test sets safely to avoid data leakage.
5. 🛠️ **Feature Engineering**: Extract datetime/cyclical features, construct domain ratios, engineer text length features, bin continuous variables, and compute group aggregations.
6. 🎯 **Feature Selection**: Remove near-zero variance features, drop high multicollinearity features, use RFE or Tree Feature Importances to select the top predictive variables.
7. 🤖 **Model Training & Evaluation**: Train your ML algorithm and measure metrics (MSE, $R^2$, Accuracy, Precision, Recall, F1-Score, ROC-AUC)!
