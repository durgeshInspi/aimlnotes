# What is Pandas?

**Pandas** is an open-source Python library used for:

* Data Analysis
* Data Cleaning
* Data Manipulation
* Data Processing
* Reading/Writing files
* Preparing data for Machine Learning

It is one of the most important libraries in Python.

---

# Why Pandas?

Without Pandas

```python
students = [
    ["Rahul",20,85],
    ["Priya",21,90],
    ["Amit",19,78]
]
```

Searching, filtering, sorting, updating is difficult.

With Pandas

```python
import pandas as pd

df = pd.DataFrame({
    "Name":["Rahul","Priya","Amit"],
    "Age":[20,21,19],
    "Marks":[85,90,78]
})

print(df)
```

Everything becomes easy.

---

# Features of Pandas

* Fast
* Easy Syntax
* Powerful
* Data Cleaning
* Missing Value Handling
* Data Filtering
* Data Sorting
* Grouping
* Statistical Analysis
* Merge Multiple Files
* CSV Support
* Excel Support
* JSON Support

---

# Installation

```bash
pip install pandas
```

Check Version

```python
import pandas as pd

print(pd.__version__)
```

---

# Import Pandas

```python
import pandas as pd
```

---

# Pandas Data Structures

Pandas has two main data structures.

1. Series (1-D)
2. DataFrame (2-D)

---

# Series

A Series is a one-dimensional array.

```python
import pandas as pd

s = pd.Series([10,20,30,40])

print(s)
```

Output

```
0   10
1   20
2   30
3   40
```

---

Series with Custom Index

```python
s = pd.Series([100,200,300],index=["A","B","C"])

print(s)
```

Access

```python
print(s["A"])
```

---

# DataFrame

A DataFrame is a table consisting of rows and columns.

```python
import pandas as pd

data = {
    "Name":["Rahul","Priya","Amit"],
    "Age":[20,21,19],
    "Marks":[85,90,78]
}

df = pd.DataFrame(data)

print(df)
```

---

# Creating DataFrame

Dictionary

```python
df = pd.DataFrame({
    "Name":["Rahul","Priya"],
    "Age":[20,21]
})
```

List

```python
data = [
    ["Rahul",20],
    ["Priya",21]
]

df = pd.DataFrame(data,columns=["Name","Age"])
```

---

# Read CSV

```python
df = pd.read_csv("students.csv")
```

---

# Read Excel

```python
df = pd.read_excel("students.xlsx")
```

Install

```bash
pip install openpyxl
```

---

# Read JSON

```python
df = pd.read_json("student.json")
```

---

# Save CSV

```python
df.to_csv("output.csv")
```

Without Index

```python
df.to_csv("output.csv",index=False)
```

---

# Save Excel

```python
df.to_excel("output.xlsx",index=False)
```

---

# Data Inspection

First Rows

```python
df.head()
```

First 10 Rows

```python
df.head(10)
```

Last Rows

```python
df.tail()
```

Information

```python
df.info()
```

Shape

```python
df.shape
```

Columns

```python
df.columns
```

Index

```python
df.index
```

Describe

```python
df.describe()
```

Data Types

```python
df.dtypes
```

---

# Selecting Columns

```python
df["Name"]
```

Multiple Columns

```python
df[["Name","Marks"]]
```

---

# Selecting Rows

Using loc

```python
df.loc[0]
```

Multiple

```python
df.loc[0:3]
```

Using iloc

```python
df.iloc[0]
```

---

# Filtering

```python
df[df["Marks"]>80]
```

Age >20

```python
df[df["Age"]>20]
```

Multiple Conditions

```python
df[(df["Age"]>20) & (df["Marks"]>80)]
```

OR

```python
df[(df["Age"]>20) | (df["Marks"]>80)]
```

---

# Sorting

Ascending

```python
df.sort_values("Marks")
```

Descending

```python
df.sort_values("Marks",ascending=False)
```

Multiple

```python
df.sort_values(["Age","Marks"])
```

---

# Add Column

```python
df["City"]=["Lucknow","Delhi","Kanpur"]
```

Calculated Column

```python
df["Percentage"]=df["Marks"]/100*100
```

---

# Rename Column

```python
df.rename(columns={"Marks":"Score"})
```

---

# Update Data

```python
df.loc[0,"Marks"]=95
```

---

# Delete Column

```python
df.drop("Age",axis=1)
```

---

# Delete Row

```python
df.drop(0)
```

---

# Missing Values

Check

```python
df.isnull()
```

Count

```python
df.isnull().sum()
```

Fill

```python
df.fillna(0)
```

Fill Mean

```python
df["Marks"]=df["Marks"].fillna(df["Marks"].mean())
```

Remove Missing

```python
df.dropna()
```

---

# Duplicate Data

Check

```python
df.duplicated()
```

Remove

```python
df.drop_duplicates()
```

---

# String Functions

Upper

```python
df["Name"].str.upper()
```

Lower

```python
df["Name"].str.lower()
```

Length

```python
df["Name"].str.len()
```

Contains

```python
df["Name"].str.contains("a")
```

Replace

```python
df["Name"].str.replace("Rahul","Rohan")
```

---

# Date & Time

```python
df["Date"]=pd.to_datetime(df["Date"])
```

Year

```python
df["Date"].dt.year
```

Month

```python
df["Date"].dt.month
```

Day

```python
df["Date"].dt.day
```

Weekday

```python
df["Date"].dt.day_name()
```

---

# GroupBy

```python
df.groupby("City")
```

Average

```python
df.groupby("City")["Marks"].mean()
```

Maximum

```python
df.groupby("City")["Marks"].max()
```

Count

```python
df.groupby("City")["Name"].count()
```

---

# Aggregation

```python
df["Marks"].sum()

df["Marks"].mean()

df["Marks"].max()

df["Marks"].min()

df["Marks"].count()

df["Marks"].std()
```

---

# Merge

```python
pd.merge(df1,df2,on="ID")
```

Left Merge

```python
pd.merge(df1,df2,on="ID",how="left")
```

Right Merge

```python
pd.merge(df1,df2,on="ID",how="right")
```

Inner Merge

```python
pd.merge(df1,df2,on="ID",how="inner")
```

Outer Merge

```python
pd.merge(df1,df2,on="ID",how="outer")
```

---

# Join

```python
df1.join(df2)
```

---

# Concatenate

```python
pd.concat([df1,df2])
```

Column Wise

```python
pd.concat([df1,df2],axis=1)
```

---

# Pivot Table

```python
pd.pivot_table(
    df,
    values="Marks",
    index="City",
    aggfunc="mean"
)
```

---

# Crosstab

```python
pd.crosstab(df["City"],df["Course"])
```

---

# Apply Function

```python
df["Marks"].apply(lambda x:x+5)
```

---

# Map

```python
df["Gender"]=df["Gender"].map({
    "M":"Male",
    "F":"Female"
})
```

---

# Replace

```python
df.replace("Lucknow","Delhi")
```

---

# Lambda

```python
df["Result"]=df["Marks"].apply(
    lambda x:"Pass" if x>=40 else "Fail"
)
```

---

# Value Counts

```python
df["City"].value_counts()
```

---

# Unique Values

```python
df["City"].unique()
```

Number of Unique Values

```python
df["City"].nunique()
```

---

# Statistical Functions

```python
df.mean(numeric_only=True)

df.median(numeric_only=True)

df.mode()

df.var(numeric_only=True)

df.std(numeric_only=True)

df.quantile(0.25)
```

---

# Correlation

```python
df.corr(numeric_only=True)
```

---

# Boolean Indexing

```python
df[df["Marks"]>=90]
```

```python
df[df["City"]=="Lucknow"]
```

---

# isin()

```python
df[df["City"].isin(["Lucknow","Delhi"])]
```

---

# between()

```python
df[df["Marks"].between(60,90)]
```

---

# Query

```python
df.query("Marks>80")
```

---

# Sample

Random Rows

```python
df.sample(5)
```

---

# Index

Set Index

```python
df.set_index("ID")
```

Reset Index

```python
df.reset_index()
```

---

# Iterating

```python
for index,row in df.iterrows():
    print(index,row["Name"])
```

---

# Export JSON

```python
df.to_json("student.json")
```

---

# Visualization

```python
import matplotlib.pyplot as plt

df["Marks"].plot()

plt.show()
```

Bar Graph

```python
df.plot(kind="bar")
```

Histogram

```python
df["Marks"].plot(kind="hist")
```

Pie Chart

```python
df["Marks"].plot(kind="pie")
```

---

# Real Project

## Student Result Analysis

Dataset

| Roll | Name  | Course | City    | Marks |
| ---- | ----- | ------ | ------- | ----- |
| 101  | Rahul | Python | Lucknow | 85    |
| 102  | Priya | AI     | Delhi   | 91    |
| 103  | Amit  | Java   | Kanpur  | 72    |

Tasks

* Load CSV
* Check missing values
* Remove duplicates
* Highest Marks
* Lowest Marks
* Average Marks
* Pass/Fail
* Course Wise Average
* City Wise Average
* Export Final Report

---

# Mini Projects

1. Student Management Analysis
2. Employee Salary Analysis
3. Sales Dashboard
4. Hospital Record Analysis
5. IPL Data Analysis
6. Netflix Dataset Analysis
7. COVID Data Analysis
8. Weather Data Analysis
9. E-Commerce Analysis
10. Banking Transaction Analysis

---

# Pandas Interview Questions

### Beginner

1. What is Pandas?
2. Difference between Series and DataFrame?
3. How do you read a CSV file?
4. What is `head()`?
5. What is `tail()`?
6. Difference between `loc` and `iloc`?
7. What is `shape`?
8. What is `info()`?
9. What is `describe()`?
10. How do you add a new column?

---

### Intermediate

1. Difference between Merge and Join?
2. Difference between `map()` and `apply()`?
3. What is GroupBy?
4. What is Pivot Table?
5. How do you remove duplicates?
6. How do you fill missing values?
7. Difference between `replace()` and `map()`?
8. Explain boolean indexing.
9. What is correlation?
10. What is aggregation?

---

### Advanced

1. Explain MultiIndex.
2. What is Vectorization?
3. Why is Pandas faster than loops?
4. Explain `query()`.
5. Difference between `loc`, `iloc`, `at`, and `iat`.
6. Explain `groupby().agg()`.
7. What is `transform()`?
8. Difference between `merge()`, `concat()`, and `join()`.
9. What is a Pivot Table?
10. Explain memory optimization in Pandas.

---

# Practice Questions

### Basic

1. Create a Series of 10 numbers.
2. Create a DataFrame of students.
3. Display the first 5 rows.
4. Display the last 3 rows.
5. Print column names.
6. Print data types.
7. Select only the Name column.
8. Select Name and Marks columns.
9. Add a City column.
10. Delete the Age column.

---

### Intermediate

1. Filter students with Marks > 80.
2. Sort students by Marks.
3. Replace missing values with 0.
4. Remove duplicate rows.
5. Find average marks.
6. Group by City.
7. Group by Course.
8. Find highest marks in each city.
9. Rename columns.
10. Save the DataFrame to a CSV file.

---

### Advanced

1. Merge two DataFrames.
2. Create a Pivot Table.
3. Create a Crosstab.
4. Convert Date column to datetime.
5. Extract year and month.
6. Apply a custom function using `apply()`.
7. Use `map()` for categorical values.
8. Calculate correlation.
9. Perform statistical analysis.
10. Build a Student Result Analysis Dashboard.

---

# Summary

After completing this guide, students will be able to:

* ✔ Understand Series and DataFrame
* ✔ Read and write CSV, Excel, and JSON files
* ✔ Inspect and clean data
* ✔ Handle missing and duplicate values
* ✔ Filter, sort, and update data
* ✔ Perform statistical analysis
* ✔ Group and aggregate data
* ✔ Merge, join, and concatenate DataFrames
* ✔ Create pivot tables and crosstabs
* ✔ Work with strings and dates
* ✔ Visualize data using Matplotlib
* ✔ Build real-world data analysis projects
* ✔ Prepare data for Machine Learning

---

