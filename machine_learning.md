# Complete Machine Learning Notes for Students

---

## 📌 Master Table of Contents
1. [Module 1: Introduction to Machine Learning](#module-1-introduction-to-machine-learning)
   - What is Machine Learning?
   - Why Do We Use Machine Learning?
   - Types of Machine Learning (Supervised, Unsupervised, Semi-Supervised, Reinforcement) with Real Examples
2. [Module 2: Exploratory Data Analysis (EDA)](#module-2-exploratory-data-analysis-eda)
   - What is EDA?
   - Why is EDA Important?
   - Every Step of EDA (Step-by-Step Pipeline with Python Code)
3. [Module 3: Data Cleaning](#module-3-data-cleaning)
   - What is Data Cleaning?
   - Handling Missing Values
   - Handling Duplicate Data
   - Outlier Detection & Treatment
   - Handling Inconsistent & Corrupted Data
4. [Module 4: Data Preprocessing](#module-4-data-preprocessing)
   - What is Data Preprocessing?
   - Feature Scaling & Normalization (StandardScaler, MinMaxScaler, RobustScaler)
   - Encoding Categorical Data (One-Hot, Ordinal, Label Encoding)
   - Train-Test Split & Preventing Data Leakage
5. [Module 5: Feature Engineering](#module-5-feature-engineering)
   - What is Feature Engineering?
   - Why Feature Engineering is Essential?
   - Feature Transformations (Log, Power, Polynomial)
   - Feature Creation & Domain Knowledge Extraction
   - Feature Binning / Discretization
6. [Module 6: Feature Selection](#module-6-feature-selection)
   - What is Feature Selection?
   - Why Feature Selection is Important?
   - Filter Methods (Variance, Correlation, Chi-Square, ANOVA, Mutual Info)
   - Wrapper Methods (Forward Selection, Backward Elimination, RFE)
   - Embedded Methods (Lasso L1 Regularization, Tree Importance)

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
  - **House Price Prediction**: Inputs: area (sq ft), bedrooms, location $\longrightarrow$ Target: Price ($ \$350,000 $).
  - **Stock Market Price Forecasting**: Inputs: past trend, volume, interest rate $\longrightarrow$ Target: Tomorrow's stock price.
  - **Salary Prediction**: Inputs: years of experience, education level $\longrightarrow$ Target: Salary ($ \$85,000 $).

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

**Exploratory Data Analysis (EDA)** is the critical initial process of performing investigations on data to summarize main characteristics, uncover underlying structures, detect anomalies/outliers, test hypotheses, and check mathematical assumptions using summary statistics and graphical visualizations.

> **Analogy**: EDA is like a detective inspecting a crime scene or a doctor examining a patient before diagnosing or writing a prescription!

---

## 2.2 Why is EDA Important?

1. **Gives Understanding of Raw Data**: Helps understand what features exist, their formats, and how they relate.
2. **Spots Errors & Missing Values**: Prevents "Garbage In, Garbage Out" in Machine Learning algorithms.
3. **Identifies Outliers**: Detects extreme values that could distort ML model predictions.
4. **Reveals Relationships & Patterns**: Uncovers correlation between independent variables and target variables.
5. **Guides Feature Selection & Model Choice**: Helps decide whether to use linear models, tree-based models, or neural networks based on data distributions.

---

## 2.3 Every Step of EDA (Step-by-Step Pipeline with Python Code)

A complete professional EDA workflow consists of **6 systematic steps**:

```
Step 1: Data Understanding & Profiling
   |
Step 2: Missing Value Identification
   |
Step 3: Univariate Analysis (Single Feature)
   |
Step 4: Bivariate & Multivariate Analysis (Multiple Features)
   |
Step 5: Outlier Detection & Analysis
   |
Step 6: Skewness & Distribution Assessment
```

---

### Step 1: Data Understanding & Profiling

Inspect data shape, column names, data types, and initial rows.

```python
import pandas as pd
import numpy as np

# Load Sample Dataset
df = pd.read_csv("student_data.csv")

# 1. View first 5 rows
print(df.head())

# 2. Check total rows and columns
print("Data Shape (Rows, Cols):", df.shape)

# 3. Check Data Types & Memory Usage
print(df.info())

# 4. Statistical Summary of Numerical Columns
print(df.describe())

# 5. Statistical Summary of Categorical Columns
print(df.describe(include='object'))
```

---

### Step 2: Missing Value Identification

Check which columns contain missing (`NaN` / `null`) values.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Check missing count and percentage per column
missing_count = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing Count': missing_count,
    'Percentage (%)': missing_percent
})

print(missing_df[missing_df['Missing Count'] > 0])

# Visualize Missing Values using Heatmap
plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()
```

---

### Step 3: Univariate Analysis (Analyzing One Feature at a Time)

Univariate analysis inspects each feature individually to understand its distribution, frequency, and central tendency.

#### A. Numerical Variables (Histogram, KDE Plot, Box Plot)
```python
# Distribution of Age feature
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.histplot(df['Age'], kde=True, color='skyblue')
plt.title("Age Distribution (Histogram + KDE)")

plt.subplot(1, 2, 2)
sns.boxplot(y=df['Age'], color='lightgreen')
plt.title("Age Boxplot (Check Spread)")

plt.show()
```

#### B. Categorical Variables (Count Plot, Bar Chart)
```python
# Distribution of Gender or Education Level
plt.figure(figsize=(6, 4))
sns.countplot(x='Gender', data=df, palette='Set2')
plt.title("Gender Frequency Distribution")
plt.show()
```

---

### Step 4: Bivariate & Multivariate Analysis (Analyzing Relationships Between Features)

Bivariate analysis explores how two features interact (e.g., Feature vs Target).

#### A. Numerical vs Numerical (Scatter Plot, Correlation Heatmap)
```python
# Scatter Plot: Study Hours vs Exam Score
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Study_Hours', y='Exam_Score', data=df, hue='Passed')
plt.title("Study Hours vs Exam Score")
plt.show()

# Correlation Matrix Heatmap
plt.figure(figsize=(8, 6))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
```

#### B. Categorical vs Numerical (Grouped Boxplot, Groupby Aggregation)
```python
# Exam Score by Education Level
plt.figure(figsize=(8, 5))
sns.boxplot(x='Education_Level', y='Exam_Score', data=df)
plt.title("Exam Score Across Education Levels")
plt.show()

# Groupby Aggregation
print(df.groupby('Gender')['Exam_Score'].mean())
```

---

### Step 5: Outlier Detection & Analysis

Outliers are extreme values that deviate significantly from the rest of the observations.

#### A. Detecting Outliers using Boxplots & IQR Method
$$\text{IQR} = Q3 - Q1$$
$$\text{Lower Bound} = Q1 - 1.5 \times \text{IQR}$$
$$\text{Upper Bound} = Q3 + 1.5 \times \text{IQR}$$

```python
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Salary'] < lower_bound) | (df['Salary'] > upper_bound)]
print(f"Total Outliers Detected: {len(outliers)}")
```

---

### Step 6: Skewness & Distribution Assessment

Skewness measures symmetry in data distribution.

```python
# Check Skewness score for numerical features
skewness = df.skew(numeric_only=True)
print(skewness)
```
- **Skewness = 0**: Perfect normal (Gaussian) distribution (Bell curve).
- **Positive Skewness (> 0.5)**: Tail extends to the right (e.g., Income data).
- **Negative Skewness (< -0.5)**: Tail extends to the left (e.g., Age of retirement).

---
---

# Module 3: Data Cleaning

## 3.1 What is Data Cleaning?

**Data Cleaning** (also known as Data Cleansing or Data Wrangling) is the process of detecting and correcting (or removing) corrupt, inaccurate, incomplete, improperly formatted, or duplicate records from a raw dataset.

> **Rule of Thumb**: Data Cleaning deals with **fixing incorrect or missing data**, whereas Data Preprocessing deals with **preparing valid data for ML algorithms**.

---

## 3.2 Handling Missing Values

Missing values occur due to data corruption, unrecorded responses, or system faults.

### Types of Missing Data:
1. **MCAR (Missing Completely at Random)**: Missingness has no pattern or relation to any variable.
2. **MAR (Missing at Random)**: Missingness depends on observed features (e.g., women tend not to answer weight questions).
3. **MNAR (Missing Not at Random)**: Missingness depends on unobserved values (e.g., people with low income skip reporting income).

### Handling Strategies:

```
                          +------------------------+
                          | Handling Missing Data  |
                          +-----------+------------+
                                      |
                 +--------------------+--------------------+
                 |                                         |
        +--------v-------+                        +--------v-------+
        |  Deletion      |                        |  Imputation    |
        |  (Drop Rows)   |                        |  (Fill Values) |
        +----------------+                        +----------------+
```

#### Strategy 1: Dropping Missing Values (`dropna`)
*Use when*: Missing data is less than 5% of total dataset, or a column has >50% missing values.

```python
# Drop rows where any column has missing value
df_clean = df.dropna()

# Drop column if more than 50% values are missing
df_clean = df.drop(columns=['Unnecessary_Column'])
```

#### Strategy 2: Statistical Imputation (Mean / Median / Mode)
- **Mean Imputation**: Use for numerical columns with **Normal Distribution** (no outliers).
- **Median Imputation**: Use for numerical columns with **Skewed Data / Outliers**.
- **Mode Imputation**: Use for **Categorical Columns**.

```python
# Numerical Imputation (Median for Skewed data)
median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)

# Categorical Imputation (Mode)
mode_gender = df['Gender'].mode()[0]
df['Gender'].fillna(mode_gender, inplace=True)
```

#### Strategy 3: Advanced Machine Learning Imputation (`KNNImputer`)
Uses k-Nearest Neighbors to predict missing values based on neighboring rows.

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)
```

---

## 3.3 Handling Duplicate Data

Duplicate rows waste computational memory and cause data bias.

```python
# Check count of duplicate rows
print("Duplicate Rows Count:", df.duplicated().sum())

# Drop Duplicate Rows
df.drop_duplicates(inplace=True)
```

---

## 3.4 Outlier Detection & Treatment

Outliers can distort algorithms like Linear Regression, Logistic Regression, and K-Means.

### Outlier Treatment Methods:

#### Method 1: Trimming / Dropping
Remove rows containing extreme outliers.

#### Method 2: Capping / Winsorization (IQR Method)
Cap extreme values at upper and lower boundary thresholds.

```python
# Capping using IQR
Q1 = df['Income'].quantile(0.25)
Q3 = df['Income'].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Cap values outside boundaries
df['Income'] = np.where(df['Income'] > upper_limit, upper_limit, df['Income'])
df['Income'] = np.where(df['Income'] < lower_limit, lower_limit, df['Income'])
```

#### Method 3: Log Transformation
Compresses large outlier values into smaller ranges.

```python
df['Income_Log'] = np.log1p(df['Income'])
```

---

## 3.5 Handling Inconsistent & Corrupted Data

Raw datasets often have typos, leading/trailing whitespace, inconsistent capitalizations, or wrong data types.

```python
# 1. Strip Whitespace and Convert String to Lowercase
df['City'] = df['City'].str.strip().str.lower()

# 2. Fix Incorrect / Inconsistent Categorical Labels
df['Gender'] = df['Gender'].replace({'M': 'Male', 'F': 'Female', 'femal': 'Female'})

# 3. Convert Data Types (.astype)
df['Age'] = df['Age'].astype(int)
df['Joining_Date'] = pd.to_datetime(df['Joining_Date'])
```

---
---

# Module 4: Data Preprocessing

## 4.1 What is Data Preprocessing?

**Data Preprocessing** transforms clean raw data into an optimized, mathematically consistent format ready for Machine Learning model training.

Key Preprocessing Tasks:
1. **Feature Scaling & Normalization**
2. **Encoding Categorical Data**
3. **Train-Test Splitting**

---

## 4.2 Feature Scaling & Normalization

Machine Learning models based on distance metrics (KNN, SVM, K-Means, Gradient Descent models like Linear/Logistic Regression) perform poorly when numerical features have wildly different scales (e.g., `Age` range 0–100 vs `Salary` range 10,000–500,000).

```
Without Scaling: Salary (50,000) completely dominates Age (25) in distance math!
With Scaling   : Both Age and Salary are brought to comparable numerical scales.
```

---

### Scaling Techniques Comparison

#### 1. Standardization (`StandardScaler`)
Transforms features so they have a **Mean ($\mu$) of 0** and **Standard Deviation ($\sigma$) of 1**.
$$Z = \frac{X - \mu}{\sigma}$$

- **When to use**: Standard scaling is best for algorithms assuming normal distribution (Linear Regression, Logistic Regression, SVM, KNN, PCA).

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['Age', 'Salary']])
```

#### 2. Min-Max Normalization (`MinMaxScaler`)
Scales features strictly within a bounded range, typically **[0, 1]**.
$$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

- **When to use**: When data does not follow Gaussian distribution, or for Neural Networks & Computer Vision images (pixel values 0–255 scaled to 0–1).

```python
from sklearn.preprocessing import MinMaxScaler

minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X[['Age', 'Salary']])
```

#### 3. Robust Scaling (`RobustScaler`)
Scales data using the **Median** and **Interquartile Range (IQR)**.
$$\text{Scaled} = \frac{X - \text{Median}}{\text{IQR}}$$

- **When to use**: When dataset contains significant **outliers** that cannot be deleted.

```python
from sklearn.preprocessing import RobustScaler

robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X[['Age', 'Salary']])
```

---

### Scaler Selection Guide

| Feature / Algorithm | Preferred Scaling Method |
| :--- | :--- |
| **Linear / Logistic Regression, SVM** | `StandardScaler` |
| **K-Means, KNN** | `StandardScaler` or `MinMaxScaler` |
| **Neural Networks, Image Processing** | `MinMaxScaler` (0 to 1) |
| **Data with heavy Outliers** | `RobustScaler` |
| **Tree-based Models (Decision Trees, Random Forests, XGBoost)** | **No Scaling Required!** |

---

## 4.3 Encoding Categorical Data

Machine Learning algorithms operate on numbers, not text. Categorical variables must be converted into numbers.

Categorical Data types:
- **Nominal Data**: Categories with **no order** (e.g., Color: `Red`, `Blue`, `Green`; Country: `India`, `USA`, `UK`).
- **Ordinal Data**: Categories with an **explicit rank or order** (e.g., Education: `High School` < `Bachelors` < `Masters` < `PhD`; Rating: `Low` < `Medium` < `High`).

---

### Encoding Methods:

#### Method 1: One-Hot Encoding (For Nominal Data)
Creates a new binary dummy column (0 or 1) for each unique category.

```
Original Column: Color          One-Hot Encoded:
| Color |                     | Color_Red | Color_Blue | Color_Green |
| Red   |         --->        |     1     |     0      |      0      |
| Blue  |                     |     0     |     1      |      0      |
| Green |                     |     0     |     0      |      1      |
```

```python
# Using Pandas get_dummies
df_encoded = pd.get_dummies(df, columns=['Color'], drop_first=True)

# Using Scikit-Learn OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(drop='first', sparse_output=False)
encoded_array = ohe.fit_transform(df[['Color']])
```
> **Note**: `drop_first=True` avoids the **Dummy Variable Trap** (Multicollinearity).

#### Method 2: Label / Ordinal Encoding (For Ordinal Data)
Assigns sequential integers based on natural rank order.

```python
from sklearn.preprocessing import OrdinalEncoder

# Specify explicit ranking order
education_order = [['High School', 'Bachelors', 'Masters', 'PhD']]
ordinal_enc = OrdinalEncoder(categories=education_order)

df['Education_Encoded'] = ordinal_enc.fit_transform(df[['Education_Level']])
```

---

## 4.4 Train-Test Split & Preventing Data Leakage

To evaluate model performance accurately on unseen data, split dataset into **Training Set (80%)** and **Testing Set (20%)**.

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

> ⚠️ **CRITICAL RULE to Avoid Data Leakage**:
> Always **fit** scalers/encoders only on `X_train`, then **transform** both `X_train` and `X_test`. Never fit on the full dataset!

```python
# CORRECT WAY:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and Transform on Train
X_test_scaled = scaler.transform(X_test)        # Transform ONLY on Test
```

---
---

# Module 5: Feature Engineering

## 5.1 What is Feature Engineering?

**Feature Engineering** is the process of using domain knowledge to transform raw data attributes into **new, meaningful input features** that improve the performance, accuracy, and interpretability of Machine Learning algorithms.

> **Famous Data Science Quote**: *"Coming up with features is complicated, time-consuming, requires domain knowledge. Applied machine learning is basically feature engineering."* — Prof. Andrew Ng

---

## 5.2 Why Feature Engineering is Essential

1. **Improves Model Accuracy**: Smart features often boost model performance more than tuning complex hyper-parameters.
2. **Simplifies Model Complexity**: Enables simpler linear algorithms to learn complex nonlinear patterns.
3. **Exposes Hidden Signals**: Transforms raw dates, strings, or timestamps into actionable signals.

---

## 5.3 Feature Transformation Techniques

Used to transform feature distributions to conform to linear assumptions or normalize skewness.

### 1. Log Transformation
Used for right-skewed data (e.g., Income, Revenue, House Prices). Compresses long right tails into normal curves.
$$X_{log} = \log(X + 1)$$

```python
df['Income_Log'] = np.log1p(df['Income'])
```

### 2. Polynomial Features
Creates interaction terms and higher-degree terms ($X_1^2, X_2^2, X_1 X_2$) to capture non-linear relationships.

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X[['Feature1', 'Feature2']])
```

---

## 5.4 Feature Creation & Domain Knowledge Extraction

Creating new domain-specific variables from raw data attributes.

### A. Date & Time Feature Extraction
A raw timestamp like `"2026-08-11 17:24:00"` cannot be used directly by ML. We extract individual features:

```python
df['Date'] = pd.to_datetime(df['Timestamp'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
df['Hour'] = df['Date'].dt.hour
```

### B. Domain Ratios & Combinations
Creating intuitive ratios from existing columns:

```python
# 1. Total Family Income in Loan Dataset
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']

# 2. Debt-to-Income Ratio
df['Debt_Income_Ratio'] = df['Total_Debt'] / (df['Total_Income'] + 1)

# 3. Price per Square Foot in Housing Dataset
df['Price_Per_SqFt'] = df['House_Price'] / df['Total_SqFt']
```

---

## 5.5 Feature Discretization / Binning

Binning converts continuous numerical numbers into categorical intervals (bins).

- **Fixed-Width Binning (`pd.cut`)**: Bins based on explicit numerical boundaries.
- **Quantile Binning (`pd.qcut`)**: Bins based on percentiles (equal count in each bin).

```python
# Binning Age into Categories
bins = [0, 18, 35, 60, 100]
labels = ['Child', 'Young Adult', 'Middle Aged', 'Senior']

df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
```

---
---

# Module 6: Feature Selection

## 6.1 What is Feature Selection?

**Feature Selection** is the process of selecting a subset of the most relevant features (columns) for use in model construction, while dropping redundant, noisy, or irrelevant features.

> **Feature Selection vs Dimensionality Reduction**:
> - **Feature Selection**: Keeps a subset of original features without altering their meaning (e.g., selecting 10 columns out of 50).
> - **Dimensionality Reduction (PCA)**: Combines/transforms features into brand new synthesized components.

---

## 6.2 Why Feature Selection is Important?

1. **Avoids Curse of Dimensionality**: Too many features lead to sparse data matrices and overfitting.
2. **Reduces Training Time**: Fewer features mean faster model training and inference speed.
3. **Enhances Model Interpretability**: Easier to explain model predictions to stakeholders.
4. **Prevents Overfitting**: Removing noisy features improves generalization on unseen test data.

---

## 6.3 Categories of Feature Selection Methods

There are **3 primary methods** for Feature Selection:

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

### 1. Filter Methods (Fast & Statistical)

Filter methods evaluate the statistical relationship between each input feature and the target variable **independent of any ML model**.

#### A. Variance Threshold
Removes features with low variance (features that stay almost constant across all rows).

```python
from sklearn.feature_selection import VarianceThreshold

# Remove columns with variance lower than 0.1
selector = VarianceThreshold(threshold=0.1)
X_high_var = selector.fit_transform(X)
```

#### B. Correlation Matrix (Removing Multicollinearity)
If two independent features are 95%+ correlated, remove one of them to prevent redundancy.

```python
corr_matrix = df.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Drop features with correlation > 0.85
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]
df_selected = df.drop(columns=to_drop)
```

#### C. Statistical Tests ($\chi^2$, ANOVA, Mutual Information)
- **Chi-Square ($\chi^2$)**: For Categorical Feature vs Categorical Target.
- **ANOVA F-Test**: For Numerical Feature vs Categorical Target.
- **Mutual Information**: Measures non-linear dependency between features and target.

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Select top 5 features using ANOVA F-test
selector = SelectKBest(score_func=f_classif, k=5)
X_top5 = selector.fit_transform(X, y)
```

---

### 2. Wrapper Methods (Search & Iterate)

Wrapper methods use an ML model as a evaluator to search for the best-performing subset of features. They test combinations iteratively.

#### A. Forward Feature Selection
Starts with 0 features and adds the best-performing feature one by one until performance stops improving.

#### B. Backward Feature Elimination
Starts with all features and removes the least significant feature step-by-step.

#### C. Recursive Feature Elimination (RFE)
Iteratively fits an estimator and removes features with the lowest weights/importance score.

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
rfe = RFE(estimator=model, n_features_to_select=5)
rfe.fit(X_train, y_train)

# View selected features boolean mask
print("Selected Features Mask:", rfe.support_)
print("Feature Ranking:", rfe.ranking_)
```

---

### 3. Embedded Methods (Built-in Regularization / Importance)

Embedded methods perform feature selection automatically during the model training process.

#### A. Lasso Regression (L1 Regularization)
Lasso adds an L1 penalty to the loss function that shrinks non-essential feature coefficients to **exact zero**, effectively dropping them.

$$\text{Loss} = \text{MSE} + \lambda \sum |\beta_j|$$

```python
from sklearn.linear_model import LassoCV

lasso = LassoCV(cv=5).fit(X_train, y_train)
important_features = X.columns[lasso.coef_ != 0]
print("Lasso Selected Features:", important_features)
```

#### B. Tree-Based Feature Importances
Decision Trees, Random Forests, and XGBoost naturally calculate feature importance based on mean decrease in impurity (Gini Impurity / Entropy).

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier().fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X.columns)

# Plot top 10 important features
importances.nlargest(10).plot(kind='barh', color='teal')
plt.title("Random Forest Top 10 Feature Importances")
plt.show()
```

---

## 📌 Feature Selection Method Decision Matrix

| Method | Speed | Model Dependent? | Detects Interactions? | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Filter Methods** | Very Fast | ❌ No | ❌ No | Initial quick filtering on huge datasets |
| **Wrapper Methods (RFE)** | Slow | ✅ Yes | ✅ Yes | Small datasets requiring peak accuracy |
| **Embedded (Lasso/Trees)** | Fast/Medium | ✅ Yes | ✅ Yes | Modern ML pipeline standard practice |

---

## 🎓 Summary Checklist for Students
1. **Understand Problem & ML Type** (Supervised vs Unsupervised).
2. **Perform EDA** (Understand shapes, distributions, anomalies).
3. **Clean Data** (Fix missing values, duplicates, outliers).
4. **Preprocess Data** (Scale numbers, encode categories, split train/test).
5. **Engineer Features** (Transform skewness, create ratios, bin columns).
6. **Select Best Features** (Drop redundant/noisy variables).
7. **Train & Evaluate ML Model!**
