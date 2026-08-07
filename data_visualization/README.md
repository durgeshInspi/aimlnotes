# 📊 Data Visualization using Matplotlib & Seaborn

Welcome to the ultimate classroom guide for **Data Visualization in Python** using **Matplotlib** and **Seaborn**. This document is designed as a complete 2-hour interactive teaching module suitable for beginner to intermediate students who are already familiar with Python basics, NumPy, and Pandas.

---

## 📚 Table of Contents
1. [Introduction](#1-introduction)
2. [Python Libraries](#2-python-libraries)
3. [Installation](#3-installation)
4. [Sample Dataset](#4-sample-dataset)
5. [Matplotlib Introduction](#5-matplotlib-introduction)
6. [Important Matplotlib Functions](#6-important-matplotlib-functions)
7. [Line Plot](#7-line-plot)
8. [Bar Chart](#8-bar-chart)
9. [Scatter Plot](#9-scatter-plot)
10. [Histogram](#10-histogram)
11. [Pie Chart](#11-pie-chart)
12. [Box Plot](#12-box-plot)
13. [Chart Customization](#13-chart-customization)
14. [Multiple Charts](#14-multiple-charts)
15. [Seaborn Introduction](#15-seaborn-introduction)
16. [Seaborn Built-in Dataset](#16-seaborn-built-in-dataset)
17. [Important Seaborn Functions](#17-important-seaborn-functions)
18. [Difference Table](#18-difference-table)
19. [Common Errors](#19-common-errors)
20. [Best Practices](#20-best-practices)
21. [Practice Questions](#21-practice-questions)
22. [Assignments](#22-assignments)
23. [Interview Questions](#23-interview-questions)
24. [Summary](#24-summary)

---

## 1. Introduction

### What is Data Visualization?
**Data Visualization** is the graphical representation of data and information using visual elements such as charts, graphs, plots, maps, and diagrams. It translates raw numbers and complex datasets into visual formats that human minds can process quickly and intuitively.

> 💡 **Trainer Note:** Humans digest visual information 60,000 times faster than text or raw tabular data. When presenting to business stakeholders, raw numbers confuse, but clear charts persuade.

---

### Why is Data Visualization Important?
1. **Simplifies Complex Data:** Enables rapid comprehension of large datasets.
2. **Identifies Trends & Patterns:** Highlights linear growth, seasonality, and clusters.
3. **Detects Anomalies & Outliers:** Makes unexpected data points (errors or fraud) immediately visible.
4. **Accelerates Decision Making:** Helps managers make data-backed business decisions without reading hundreds of rows.
5. **Storytelling with Data:** Transforms cold analytics into a compelling story for non-technical audiences.

---

### Advantages of Data Visualization
* **Clarity:** Raw rows of data become clean visual signals.
* **Efficiency:** Analyzes millions of data points in seconds visually.
* **Engagement:** Visual representations keep audiences interested during reports and presentations.
* **Comparison:** Easily compares multiple metrics side-by-side (e.g., Sales vs Revenue across quarters).

---

### Real-World Applications
| Industry | Application Example |
| :--- | :--- |
| **Healthcare** | Tracking patient recovery rates, disease spread maps, and heart rate monitoring over time. |
| **Finance** | Stock market price trend tracking, fraud detection anomaly graphs, risk assessment portfolios. |
| **E-Commerce** | Visualizing customer funnel drop-offs, sales spikes during holiday seasons, inventory demand. |
| **Marketing** | Social media campaign engagement heatmaps, customer demographic distributions. |
| **Weather Forecasting** | Temperature trend lines, rainfall histograms, cyclone movement trajectories. |

---

### Types of Charts & When to Use Each

| Chart Type | Best Used For | Example Use Case |
| :--- | :--- | :--- |
| **Line Plot** | Continuous data over time (Time Series) | Stock prices over 12 months |
| **Bar Chart** | Categorical data comparison | Product sales across 5 categories |
| **Scatter Plot** | Relationship / Correlation between 2 numerical variables | Height vs Weight of students |
| **Histogram** | Frequency distribution of a continuous variable | Marks distribution of 100 students |
| **Pie Chart** | Proportion of parts to a whole (percentages = 100%) | Market share of mobile brands |
| **Box Plot** | Outlier detection & statistical distribution (5-number summary) | Salary distribution across departments |
| **Heatmap** | Matrix correlation & intensity levels | Feature correlation in Machine Learning |

---

## 2. Python Libraries

In Python data science workflows, four main libraries work together sequentially:

```
+-----------------------------------------------------------+
|                     Python Data Stack                     |
+-------------------+--------------------+------------------+
|      NumPy        |      Pandas        |    Matplotlib    |
| (Fast Math Arrays)| (DataFrames/Tables)| (Base Graphics)  |
+-------------------+--------------------+------------------+
                                                 |
                                                 v
                                         +---------------+
                                         |    Seaborn    |
                                         | (Stat Graphics|
                                         +---------------+
```

### 1. NumPy (`numpy`)
* **Role:** Numerical calculation engine.
* **Why used:** Provides multidimensional arrays (`ndarray`) and mathematical operations required for generating coordinate points, random datasets, and linear space sequences.

### 2. Pandas (`pandas`)
* **Role:** Data manipulation and tabular data handling.
* **Why used:** Provides `Series` and `DataFrame` objects to load CSV/Excel files, clean data, filter rows, and group metrics before visualization.

### 3. Matplotlib (`matplotlib.pyplot`)
* **Role:** Low-level, foundation plotting library.
* **Why used:** Offers complete control over every element of a figure (axes, ticks, titles, grids, legend placement, and output file saving).

### 4. Seaborn (`seaborn`)
* **Role:** High-level statistical visualization library built on top of Matplotlib.
* **Why used:** Requires significantly less code, automatically integrates with Pandas DataFrames, and produces aesthetically pleasing charts with built-in color themes.

---

## 3. Installation

To install all required libraries, run the following commands in your terminal or Jupyter Notebook cell:

```bash
# Terminal Installation Command
pip install matplotlib seaborn pandas numpy
```

### Importing Libraries in Python

```python
# Standard Industry Aliases
import matplotlib.pyplot as plt  # Used for low-level visualization control
import seaborn as sns           # Used for statistical and high-level plots
import pandas as pd             # Used for DataFrame manipulation
import numpy as np              # Used for numerical and array operations

# Display confirmation
print("All Data Visualization libraries imported successfully!")
```

---

## 4. Sample Dataset

Before plotting any chart, we need structured datasets. Below are 4 pre-built datasets used throughout this guide.

> ❓ **Why are datasets required before visualization?**
> Visualization functions require structured numerical sequences or categorical vectors as input arguments (X and Y coordinates). Without data arrays or DataFrame columns, no geometric shapes (lines, bars, dots) can be rendered.

```python
import pandas as pd
import numpy as np

# Dataset 1: Student Marks
df_students = pd.DataFrame({
    'Student_Name': ['Rahul', 'Priya', 'Amit', 'Neha', 'Suresh', 'Ananya'],
    'Maths': [85, 92, 78, 65, 88, 95],
    'Science': [90, 88, 82, 70, 84, 98],
    'English': [78, 85, 80, 75, 90, 92]
})

# Dataset 2: Monthly Sales
df_sales = pd.DataFrame({
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    'Sales_Units': [150, 180, 210, 190, 250, 310, 290, 330, 380, 420, 500, 610],
    'Revenue_USD': [15000, 18000, 21000, 19000, 25000, 31000, 29000, 33000, 38000, 42000, 50000, 61000]
})

# Dataset 3: Age & Salary
np.random.seed(42)  # Ensures reproducible results
df_employees = pd.DataFrame({
    'Age': np.random.randint(22, 60, size=50),
    'Salary_K': np.random.randint(30, 150, size=50),
    'Experience_Years': np.random.randint(1, 35, size=50)
})

# Dataset 4: Company Revenue
df_revenue = pd.DataFrame({
    'Company': ['TechCorp', 'InnoSoft', 'DataSys', 'CloudNet', 'AiWorks'],
    'Revenue_Millions': [45.5, 78.2, 32.0, 95.4, 62.1],
    'Employees': [350, 600, 200, 850, 450]
})

print("Sample Datasets Created Successfully:")
print(df_students.head(2))
```

---

## 5. Matplotlib Introduction

### What is Matplotlib?
**Matplotlib** is the fundamental 2D plotting library for Python. Created by John D. Hunter in 2003, it was designed to replicate MATLAB's plotting capabilities in Python. Its module `matplotlib.pyplot` provides a MATLAB-like stateful interface.

### Key Features
* Highly customizable (control over axes, labels, fonts, colors, line styles).
* Supports multiple output formats (PNG, JPG, PDF, SVG).
* Integrates seamlessly with NumPy, Pandas, and SciPy.
* Serves as the graphic engine for Seaborn and Pandas plotting functions.

### Advantages & Disadvantages
| Advantages | Disadvantages |
| :--- | :--- |
| Extreme customization control over every pixel | Requires long, verbose code for complex charts |
| Huge community support and documentation | Default aesthetics look dated without styling |
| Supports 2D and basic 3D plots | Handling Pandas DataFrames directly can be clunky |

---

## 6. Important Matplotlib Functions

Here is a quick-reference list of the most essential `matplotlib.pyplot` functions:

| Function | Purpose | Basic Syntax | Key Parameters |
| :--- | :--- | :--- | :--- |
| `plt.figure()` | Creates a new canvas/figure window | `plt.figure(figsize=(8,5))` | `figsize`, `dpi`, `facecolor` |
| `plt.plot()` | Plots lines or markers | `plt.plot(x, y)` | `color`, `marker`, `linestyle`, `linewidth`, `label` |
| `plt.bar()` | Draws vertical bar chart | `plt.bar(x, height)` | `color`, `width`, `align`, `edgecolor` |
| `plt.barh()` | Draws horizontal bar chart | `plt.barh(y, width)` | `color`, `height`, `align`, `edgecolor` |
| `plt.scatter()` | Plots scatter points | `plt.scatter(x, y)` | `c`, `s`, `marker`, `alpha`, `cmap` |
| `plt.hist()` | Plots frequency histogram | `plt.hist(data)` | `bins`, `color`, `edgecolor`, `density` |
| `plt.pie()` | Draws pie chart | `plt.pie(values)` | `labels`, `autopct`, `explode`, `shadow`, `startangle` |
| `plt.boxplot()` | Draws box and whisker plot | `plt.boxplot(data)` | `vert`, `patch_artist`, `notch` |
| `plt.xlabel()` | Sets X-axis title | `plt.xlabel("Text")` | `fontsize`, `color`, `labelpad` |
| `plt.ylabel()` | Sets Y-axis title | `plt.ylabel("Text")` | `fontsize`, `color`, `labelpad` |
| `plt.title()` | Sets Chart title | `plt.title("Text")` | `fontsize`, `fontweight`, `loc` |
| `plt.legend()` | Shows legend box | `plt.legend()` | `loc`, `fontsize`, `frameon` |
| `plt.grid()` | Toggles grid lines | `plt.grid(True)` | `color`, `linestyle`, `alpha` |
| `plt.subplot()` | Adds a subplot to current figure | `plt.subplot(rows, cols, index)` | `nrows`, `ncols`, `index` |
| `plt.show()` | Displays the figure on screen | `plt.show()` | None |
| `plt.savefig()` | Saves figure to local file | `plt.savefig("filename.png")` | `dpi`, `bbox_inches` |
| `plt.xticks()` | Customizes X-axis tick locations/labels | `plt.xticks(ticks, labels)` | `rotation`, `fontsize` |
| `plt.yticks()` | Customizes Y-axis tick locations/labels | `plt.yticks(ticks, labels)` | `rotation`, `fontsize` |
| `plt.xlim()` | Sets X-axis numeric limits | `plt.xlim(min, max)` | `left`, `right` |
| `plt.ylim()` | Sets Y-axis numeric limits | `plt.ylim(min, max)` | `bottom`, `top` |

---

## 7. Line Plot

### Definition & Concept
A **Line Plot** connects discrete data points with straight line segments. It is primarily used to visualize data that changes continuously over time (Time Series analysis).

### Why Use Line Plots?
* Ideal for tracking continuous trends across chronological intervals (days, months, years).
* Compares changes over time for multiple groups simultaneously.

### Syntax
```python
plt.plot(x, y, color='blue', marker='o', linestyle='-', linewidth=2, label='Series Name')
```

---

### Code Examples (Line Plot)

```python
import matplotlib.pyplot as plt

# Data
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
sales = [10, 15, 14, 20, 24, 30]
expenses = [8, 11, 12, 15, 17, 18]

# -------------------------------------------------------------
# Example 1: Basic Line Plot
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(months, sales)  # Minimal code to draw line
plt.title("1. Basic Line Plot")
plt.show()

# -------------------------------------------------------------
# Example 2: Line Plot with Color Customization
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(months, sales, color='green')  # Using named color
plt.title("2. Colored Line Plot")
plt.show()

# -------------------------------------------------------------
# Example 3: Line Plot with Custom Markers
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(months, sales, color='red', marker='o')  # Circle marker on data points
plt.title("3. Line Plot with Markers")
plt.show()

# -------------------------------------------------------------
# Example 4: Custom Line Style & Width
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(months, sales, color='purple', linestyle='--', linewidth=3)  # Dashed line
plt.title("4. Custom Line Style & Width")
plt.show()

# -------------------------------------------------------------
# Example 5: Multiple Lines on Same Chart
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(months, sales, color='blue', marker='o', label='Sales')
plt.plot(months, expenses, color='orange', marker='s', label='Expenses')  # Second series
plt.title("5. Multiple Lines Plot")
plt.legend()
plt.show()

# -------------------------------------------------------------
# Example 6: Adding Grid Lines
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(months, sales, color='teal', marker='^')
plt.title("6. Line Plot with Grid")
plt.grid(True, linestyle=':', alpha=0.6)  # Subtle dotted grid
plt.show()

# -------------------------------------------------------------
# Example 7: Line Plot with Complete Labels & Legend
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(months, sales, color='navy', marker='D', label='Monthly Growth')
plt.xlabel("Month of Year")
plt.ylabel("Sales Volume (in Units)")
plt.title("7. Complete Annotated Line Plot")
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# -------------------------------------------------------------
# Example 8: Adjusting Figure Size & Limits
# -------------------------------------------------------------
plt.figure(figsize=(8, 4))  # Wider canvas size
plt.plot(months, sales, color='darkgreen', marker='o', linewidth=2)
plt.ylim(0, 40)             # Setting explicit Y-axis range from 0 to 40
plt.title("8. Line Plot with Explicit Y-Limits (0 to 40)")
plt.show()
```

---

## 8. Bar Chart

### Definition & Concept
A **Bar Chart** presents categorical data with rectangular bars whose heights or lengths are proportional to the values they represent.

* **Vertical Bar Chart:** Categories on X-axis, Values on Y-axis.
* **Horizontal Bar Chart:** Categories on Y-axis, Values on X-axis (great for long category names).

---

### Code Examples (Bar Chart)

```python
import matplotlib.pyplot as plt

# Sample Data
courses = ['Python', 'Java', 'C++', 'Data Science', 'Web Dev']
students = [120, 85, 60, 150, 110]
enrolled_female = [50, 30, 20, 65, 45]

# -------------------------------------------------------------
# Example 1: Basic Vertical Bar Chart
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.bar(courses, students)
plt.title("1. Basic Vertical Bar Chart")
plt.show()

# -------------------------------------------------------------
# Example 2: Horizontal Bar Chart (plt.barh)
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.barh(courses, students, color='skyblue')  # Horizontal bars
plt.title("2. Horizontal Bar Chart")
plt.show()

# -------------------------------------------------------------
# Example 3: Custom Colors per Bar
# -------------------------------------------------------------
colors_list = ['gold', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink']
plt.figure(figsize=(6, 4))
plt.bar(courses, students, color=colors_list)  # Passing color list
plt.title("3. Custom Colors Bar Chart")
plt.show()

# -------------------------------------------------------------
# Example 4: Adjusting Bar Width & Edges
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.bar(courses, students, width=0.4, color='orange', edgecolor='black', linewidth=1.5)
plt.title("4. Custom Width and Black Edge Colors")
plt.show()

# -------------------------------------------------------------
# Example 5: Grouped Bar Chart (Side-by-Side)
# -------------------------------------------------------------
import numpy as np

x_indexes = np.arange(len(courses))
width = 0.35  # Bar width offset

plt.figure(figsize=(7, 4))
plt.bar(x_indexes - width/2, students, width=width, label='Total Students', color='royalblue')
plt.bar(x_indexes + width/2, enrolled_female, width=width, label='Female Students', color='pink')

plt.xticks(ticks=x_indexes, labels=courses)  # Re-label X ticks
plt.title("5. Grouped Bar Chart")
plt.legend()
plt.show()

# -------------------------------------------------------------
# Example 6: Stacked Bar Chart
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
# First layer
plt.bar(courses, students, label='Total Students', color='lightgray')
# Second layer on top of first using bottom parameter
plt.bar(courses, enrolled_female, label='Female Students', color='purple')
plt.title("6. Stacked Bar Chart")
plt.legend()
plt.show()

# -------------------------------------------------------------
# Example 7: Adding Values Labels on Top of Bars
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
bars = plt.bar(courses, students, color='seagreen')

# Annotating text over bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 2, int(yval), ha='center', va='bottom')

plt.ylim(0, 180)
plt.title("7. Bar Chart with Data Annotations")
plt.show()

# -------------------------------------------------------------
# Example 8: Horizontal Bar Chart with Custom Order & Rotated Ticks
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.bar(courses, students, color='darkred')
plt.xticks(rotation=45)  # Rotates X-axis labels by 45 degrees
plt.title("8. Bar Chart with Rotated Tick Labels")
plt.show()
```

---

## 9. Scatter Plot

### Definition & Correlation
A **Scatter Plot** uses Cartesian coordinates to display values for two numerical variables for a set of data points.

* **Positive Correlation:** As X increases, Y increases.
* **Negative Correlation:** As X increases, Y decreases.
* **No Correlation:** Points are randomly scattered without a visible slope.

---

### Code Examples (Scatter Plot)

```python
import matplotlib.pyplot as plt
import numpy as np

# Data
age = [22, 25, 30, 35, 40, 45, 50, 55, 60]
salary = [25, 32, 45, 55, 65, 70, 85, 90, 115]
company_size = [50, 100, 250, 500, 1000, 1500, 2000, 3000, 5000]

# -------------------------------------------------------------
# Example 1: Basic Scatter Plot
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.scatter(age, salary)
plt.title("1. Basic Scatter Plot")
plt.show()

# -------------------------------------------------------------
# Example 2: Custom Marker and Color
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.scatter(age, salary, color='red', marker='^')  # Triangle marker
plt.title("3. Custom Marker and Color")
plt.show()

# -------------------------------------------------------------
# Example 3: Changing Dot Size (Bubble Effect)
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
# Sizes parameter s mapped to company_size array
plt.scatter(age, salary, s=[x/10 for x in company_size], color='blue', alpha=0.6)
plt.title("3. Bubble Chart (Dot Size = Company Size)")
plt.show()

# -------------------------------------------------------------
# Example 4: Adding Transparency (Alpha)
# -------------------------------------------------------------
np.random.seed(0)
x_large = np.random.randn(500)
y_large = np.random.randn(500)

plt.figure(figsize=(6, 4))
plt.scatter(x_large, y_large, color='darkgreen', alpha=0.3)  # Overlapping dots visible
plt.title("4. Scatter Plot with Alpha (Transparency)")
plt.show()

# -------------------------------------------------------------
# Example 5: Color-Mapping by Numerical Values (cmap)
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
scatter = plt.scatter(age, salary, c=salary, cmap='viridis', s=100)
plt.colorbar(scatter, label='Salary Scale (K)')
plt.title("5. Color-Mapped Scatter Plot")
plt.show()

# -------------------------------------------------------------
# Example 6: Multiple Groups in Scatter Plot
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.scatter(age[:5], salary[:5], color='blue', label='Junior Level', s=80)
plt.scatter(age[5:], salary[5:], color='magenta', label='Senior Level', s=80)
plt.title("6. Multiple Data Series Scatter")
plt.legend()
plt.show()

# -------------------------------------------------------------
# Example 7: Scatter Plot with Trend Line (Linear Fit)
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.scatter(age, salary, color='black', label='Employee Data')

# Fit polynomial degree 1 (Line)
m, b = np.polyfit(age, salary, 1)
plt.plot(age, [m*x + b for x in age], color='red', linewidth=2, label='Trendline')

plt.title("7. Scatter Plot with Linear Trendline")
plt.legend()
plt.show()

# -------------------------------------------------------------
# Example 8: Edge Colors and Custom Markers
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.scatter(age, salary, s=150, color='yellow', edgecolor='black', linewidth=2, marker='s')
plt.title("8. Styled Squares with Black Borders")
plt.show()
```

---

## 10. Histogram

### Definition & Concept
A **Histogram** represents the distribution of continuous numerical data by grouping numbers into bins (ranges) and plotting frequency counts for each bin.

> ⚠️ **Bar Chart vs Histogram:** A Bar Chart compares discrete categories (e.g., Apple vs Orange). A Histogram displays continuous data distributions (e.g., Age brackets: 0-10, 10-20, 20-30) without spaces between adjacent bars.

---

### Code Examples (Histogram)

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate continuous sample dataset (e.g., Exam Scores of 200 students)
np.random.seed(42)
scores = np.random.normal(loc=70, scale=10, size=200)  # Mean 70, Std Dev 10

# -------------------------------------------------------------
# Example 1: Basic Histogram
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.hist(scores)
plt.title("1. Basic Histogram (Default Bins = 10)")
plt.show()

# -------------------------------------------------------------
# Example 2: Specifying Custom Bin Count
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.hist(scores, bins=20, color='skyblue')  # 20 distinct range intervals
plt.title("2. Histogram with 20 Bins")
plt.show()

# -------------------------------------------------------------
# Example 3: Adding Edge Color for Bin Separation
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.hist(scores, bins=15, color='lightgreen', edgecolor='black', linewidth=1.2)
plt.title("3. Histogram with Black Edge Borders")
plt.show()

# -------------------------------------------------------------
# Example 4: Probability Density Histogram (density=True)
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.hist(scores, bins=15, density=True, color='orange', edgecolor='white', alpha=0.7)
plt.title("4. Density Normalized Histogram (Sum of area = 1)")
plt.ylabel("Probability Density")
plt.show()

# -------------------------------------------------------------
# Example 5: Overlaid Histograms (Comparing Two Groups)
# -------------------------------------------------------------
group_A = np.random.normal(65, 8, 150)
group_B = np.random.normal(75, 8, 150)

plt.figure(figsize=(6, 4))
plt.hist(group_A, bins=15, alpha=0.5, label='Group A (Avg: 65)', color='blue', edgecolor='black')
plt.hist(group_B, bins=15, alpha=0.5, label='Group B (Avg: 75)', color='red', edgecolor='black')
plt.title("5. Overlaid Frequency Histograms")
plt.legend()
plt.show()
```

---

## 11. Pie Chart

### Definition & Concept
A **Pie Chart** is a circular statistical graphic divided into slices to illustrate numerical proportions. The arc length (and central angle) of each slice is proportional to the quantity it represents (total sum = 100%).

---

### Code Examples (Pie Chart)

```python
import matplotlib.pyplot as plt

# Data
categories = ['Mobile', 'Laptops', 'Tablets', 'Accessories', 'Wearables']
sales_share = [40, 25, 15, 12, 8]

# -------------------------------------------------------------
# Example 1: Basic Pie Chart
# -------------------------------------------------------------
plt.figure(figsize=(5, 5))
plt.pie(sales_share, labels=categories)
plt.title("1. Basic Pie Chart")
plt.show()

# -------------------------------------------------------------
# Example 2: Adding Automatic Percentages (autopct)
# -------------------------------------------------------------
plt.figure(figsize=(5, 5))
# autopct='%1.1f%%' formats floats to 1 decimal place with % sign
plt.pie(sales_share, labels=categories, autopct='%1.1f%%', startangle=90)
plt.title("2. Pie Chart with Percentage Labels")
plt.show()

# -------------------------------------------------------------
# Example 3: Exploding Slices (Pulling Slices Out)
# -------------------------------------------------------------
explode_list = [0.1, 0, 0, 0, 0]  # Explode 1st slice (Mobile) by 10%

plt.figure(figsize=(5, 5))
plt.pie(sales_share, labels=categories, autopct='%1.1f%%', explode=explode_list, shadow=True)
plt.title("3. Exploded Pie Chart with Shadow")
plt.show()

# -------------------------------------------------------------
# Example 4: Custom Color Palette & Text Styling
# -------------------------------------------------------------
custom_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']

plt.figure(figsize=(5, 5))
plt.pie(sales_share, labels=categories, autopct='%1.1f%%', colors=custom_colors, startangle=140)
plt.title("4. Styled Custom Colors Pie Chart")
plt.show()

# -------------------------------------------------------------
# Example 5: Creating a Donut Chart
# -------------------------------------------------------------
plt.figure(figsize=(5, 5))
plt.pie(sales_share, labels=categories, autopct='%1.1f%%', startangle=90, pctdistance=0.85)

# Draw central circle to turn Pie into Donut
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title("5. Modern Donut Chart")
plt.show()
```

---

## 12. Box Plot

### Definition & Statistical Concept
A **Box Plot** (Box-and-Whisker Plot) displays the 5-number summary statistical distribution of a dataset:
1. **Minimum:** Smallest value excluding outliers ($Q1 - 1.5 \times IQR$).
2. **First Quartile ($Q1$):** 25th percentile.
3. **Median ($Q2$):** 50th percentile (Middle line inside box).
4. **Third Quartile ($Q3$):** 75th percentile.
5. **Maximum:** Largest value excluding outliers ($Q3 + 1.5 \times IQR$).
6. **Outliers:** Individual points plotted beyond whiskers.

```
       Outlier
          o

       Upper Max  +-------------------+- Maximum (Q3 + 1.5*IQR)
                  |
            Q3    +===================+-- 75th Percentile
                  |                   |
          Median  |---------+---------|-- 50th Percentile (Q2)
                  |                   |
            Q1    +===================+-- 25th Percentile
                  |
       Lower Min  +-------------------+- Minimum (Q1 - 1.5*IQR)
```

---

### Code Examples (Box Plot)

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample Data
np.random.seed(10)
dept1_salaries = np.random.normal(50, 10, 100)
dept2_salaries = np.random.normal(65, 15, 100)
dept3_salaries = np.append(np.random.normal(40, 5, 95), [110, 115, 120, 15])  # Includes outliers

# -------------------------------------------------------------
# Example 1: Basic Single Box Plot
# -------------------------------------------------------------
plt.figure(figsize=(5, 4))
plt.boxplot(dept1_salaries)
plt.title("1. Basic Single Box Plot")
plt.ylabel("Salary (in $K)")
plt.show()

# -------------------------------------------------------------
# Example 2: Comparing Multiple Groups Side-by-Side
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.boxplot([dept1_salaries, dept2_salaries, dept3_salaries], labels=['Sales', 'Engineering', 'HR'])
plt.title("2. Multi-Department Salary Distribution (With Outliers)")
plt.ylabel("Salary (in $K)")
plt.show()

# -------------------------------------------------------------
# Example 3: Horizontal Box Plot
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.boxplot(dept2_salaries, vert=False)  # vert=False turns plot horizontal
plt.title("3. Horizontal Box Plot")
plt.xlabel("Salary (in $K)")
plt.show()

# -------------------------------------------------------------
# Example 4: Box Plot with Notch and Color Fill
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
# notch=True shows confidence interval around median
plt.boxplot([dept1_salaries, dept2_salaries], notch=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='blue'))
plt.title("4. Notched Box Plot with Custom Fill Color")
plt.show()

# -------------------------------------------------------------
# Example 5: Customizing Outlier Markers (fliers)
# -------------------------------------------------------------
red_diamond = dict(markerfacecolor='r', marker='D')  # Red diamond marker for outliers

plt.figure(figsize=(6, 4))
plt.boxplot(dept3_salaries, flierprops=red_diamond)
plt.title("5. Box Plot highlighting Outliers with Red Diamonds")
plt.show()
```

---

## 13. Chart Customization

Customization transforms raw plots into production-ready dashboard elements.

### Key Customization Options
1. **Title:** `plt.title("Text", fontsize=14, fontweight='bold', color='navy')`
2. **Axis Labels:** `plt.xlabel("X Text")`, `plt.ylabel("Y Text")`
3. **Grid:** `plt.grid(True, linestyle='--', alpha=0.5, color='gray')`
4. **Legend:** `plt.legend(loc='upper right', frameon=True, shadow=True)`
5. **Figure Size:** `plt.figure(figsize=(width, height), dpi=100)`
6. **Ticks & Rotation:** `plt.xticks(rotation=45, fontsize=10)`
7. **Axis Limits:** `plt.xlim(min, max)`, `plt.ylim(min, max)`
8. **Saving Figure:** `plt.savefig("chart.png", dpi=300, bbox_inches='tight')`

---

### Executable Master Customization Example

```python
import matplotlib.pyplot as plt

# Data setup
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
revenue = [120, 150, 170, 160, 210, 240]
cost = [80, 90, 100, 95, 110, 120]

# 1. Figure Initialization with Canvas Size and DPI
plt.figure(figsize=(9, 5), dpi=100, facecolor='#f8f9fa')

# 2. Plotting Lines with Markers, Colors, Width, and Labels
plt.plot(months, revenue, color='#1f77b4', marker='o', linewidth=2.5, markersize=8, label='Monthly Revenue ($K)')
plt.plot(months, cost, color='#ff7f0e', marker='s', linewidth=2.5, markersize=8, linestyle='--', label='Operating Cost ($K)')

# 3. Titles and Labels Customization
plt.title("Q1-Q2 Financial Performance Overview", fontsize=16, fontweight='bold', pad=15, color='#333333')
plt.xlabel("Operating Month", fontsize=12, fontweight='bold', color='#333333')
plt.ylabel("Amount (USD Thousands)", fontsize=12, fontweight='bold', color='#333333')

# 4. Ticks Customization and Label Rotation
plt.xticks(fontsize=11, rotation=0)
plt.yticks(fontsize=11)

# 5. Axis Limit Boundaries
plt.ylim(50, 300)

# 6. Gridlines
plt.grid(True, linestyle=':', linewidth=1, color='gray', alpha=0.5)

# 7. Legend Placement and Frame Styling
plt.legend(loc='upper left', fontsize=11, frameon=True, facecolor='white', edgecolor='none')

# 8. Saving High-Resolution Output File
plt.savefig("financial_performance.png", dpi=300, bbox_inches='tight')

# 9. Display Figure
plt.show()
print("Figure successfully styled and saved as 'financial_performance.png'!")
```

---

## 14. Multiple Charts (Subplots)

Subplots allow you to render multiple side-by-side charts within a single figure canvas.

### Method 1: `plt.subplot(nrows, ncols, index)`

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 20, 15, 25, 30]

plt.figure(figsize=(10, 4))

# First Chart (1 row, 2 columns, 1st position)
plt.subplot(1, 2, 1)
plt.plot(x, y, color='blue', marker='o')
plt.title("Line Chart (Subplot 1)")

# Second Chart (1 row, 2 columns, 2nd position)
plt.subplot(1, 2, 2)
plt.bar(x, y, color='green')
plt.title("Bar Chart (Subplot 2)")

plt.tight_layout()  # Automatically adjusts padding between subplots
plt.show()
```

---

### Method 2: Object-Oriented `plt.subplots(nrows, ncols)`

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Creates a 2x2 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))

axes[0, 0].plot(x, y, 'r-')
axes[0, 0].set_title("Top-Left: Line")

axes[0, 1].scatter(x, y, color='blue')
axes[0, 1].set_title("Top-Right: Scatter")

axes[1, 0].bar(x, y, color='purple')
axes[1, 0].set_title("Bottom-Left: Bar")

axes[1, 1].hist(y, bins=3, color='orange')
axes[1, 1].set_title("Bottom-Right: Hist")

plt.tight_layout()
plt.show()
```

---

## 15. Seaborn Introduction

### What is Seaborn?
**Seaborn** is a Python data visualization library built on top of Matplotlib. It integrates directly with Pandas DataFrames and provides sophisticated statistical charts with minimal code.

### Why Use Seaborn?
* Stunning default visual themes and color palettes.
* Summarizes and visualizes statistical relationships automatically (e.g., confidence intervals, hue groupings).
* High-level abstractions for complex statistical plots (e.g., heatmaps, pairplots, categorical point plots).

### Matplotlib vs Seaborn Comparison
| Feature | Matplotlib | Seaborn |
| :--- | :--- | :--- |
| **Abstraction** | Low-level (Detailed control) | High-level (Quick insights) |
| **Data Format** | Requires manual parsing | Direct DataFrame column pass (`data=df`) |
| **Syntax Length** | Verbose (10+ lines for styled plots) | Concise (1 to 2 lines) |
| **Statistical Features** | Manual calculation required | Built-in aggregation & regression lines |
| **Themes** | Basic/Plain defaults | Elegant aesthetic themes built-in |

### Seaborn Styles & Themes
Seaborn includes 5 built-in preset themes: `darkgrid`, `whitegrid`, `dark`, `white`, and `ticks`.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Setting default aesthetic theme and palette
sns.set_theme(style="whitegrid", palette="muted")
```

---

## 16. Seaborn Built-in Dataset (`tips`)

Seaborn includes several built-in datasets for learning and benchmarking. The most popular dataset is **`tips`**.

```python
import seaborn as sns

# Load built-in dataset
tips = sns.load_dataset("tips")

# Display first 5 rows
print(tips.head())
```

### Dataset Column Definitions
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| **`total_bill`** | Float | Total dollar amount of restaurant bill |
| **`tip`** | Float | Dollar tip amount given to server |
| **`sex`** | Category | Gender of bill payer (`Male`, `Female`) |
| **`smoker`** | Category | Whether smoking table (`Yes`, `No`) |
| **`day`** | Category | Day of the week (`Thur`, `Fri`, `Sat`, `Sun`) |
| **`time`** | Category | Meal timeframe (`Lunch`, `Dinner`) |
| **`size`** | Integer | Number of dining guests at the table |

---

## 17. Important Seaborn Functions

Here are the 8 key Seaborn plotting functions every data scientist uses:

---

### 1. `sns.lineplot()`
* **Definition:** Draws statistical line plots with automated confidence interval shading.
* **Syntax:** `sns.lineplot(data=df, x='col_x', y='col_y', hue='col_group')`

```python
import seaborn as sns
import matplotlib.pyplot as plt

fmri = sns.load_dataset("fmri")

plt.figure(figsize=(6, 4))
# Hue automatically separates series by color and adds legend
sns.lineplot(data=fmri, x="timepoint", y="signal", hue="event", style="event")
plt.title("sns.lineplot - Brain Signal over Time")
plt.show()
```

---

### 2. `sns.barplot()`
* **Definition:** Plots aggregated mean estimates along with error bar confidence intervals.
* **Syntax:** `sns.barplot(data=df, x='cat_col', y='num_col', hue='group_col')`

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

plt.figure(figsize=(6, 4))
# Displays mean tip per day split by gender
sns.barplot(data=tips, x="day", y="tip", hue="sex", palette="Blues")
plt.title("sns.barplot - Mean Tip Amount by Day and Sex")
plt.show()
```

---

### 3. `sns.scatterplot()`
* **Definition:** Draws 2D scatter plots with automatic hue color encoding and size scaling.
* **Syntax:** `sns.scatterplot(data=df, x='col_x', y='col_y', hue='group_col', size='num_col')`

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

plt.figure(figsize=(6, 4))
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time", style="time", size="size")
plt.title("sns.scatterplot - Total Bill vs Tip")
plt.show()
```

---

### 4. `sns.countplot()`
* **Definition:** Plots category frequency counts (like a categorical histogram).
* **Syntax:** `sns.countplot(data=df, x='cat_col', hue='group_col')`

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

plt.figure(figsize=(6, 4))
sns.countplot(data=tips, x="day", hue="smoker", palette="Set2")
plt.title("sns.countplot - Dining Frequency by Day and Smoking Status")
plt.show()
```

---

### 5. `sns.histplot()`
* **Definition:** Plots univariate distributions with optional Kernel Density Estimate (KDE) curve overlay.
* **Syntax:** `sns.histplot(data=df, x='num_col', kde=True, bins=20)`

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

plt.figure(figsize=(6, 4))
sns.histplot(data=tips, x="total_bill", kde=True, color="purple", bins=15)
plt.title("sns.histplot - Total Bill Distribution with KDE Curve")
plt.show()
```

---

### 6. `sns.boxplot()`
* **Definition:** Draws statistical box plots across multiple categories effortlessly.
* **Syntax:** `sns.boxplot(data=df, x='cat_col', y='num_col', hue='group_col')`

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

plt.figure(figsize=(6, 4))
sns.boxplot(data=tips, x="day", y="total_bill", hue="smoker", palette="Set3")
plt.title("sns.boxplot - Total Bill Spread across Days")
plt.show()
```

---

### 7. `sns.heatmap()`
* **Definition:** Displays 2D matrix numerical values encoded as color intensity gradients (essential for feature correlation matrices).
* **Syntax:** `sns.heatmap(data_matrix, annot=True, cmap='coolwarm')`

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")
# Calculate correlation matrix for numeric columns
numeric_tips = tips.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_tips.corr()

plt.figure(figsize=(6, 4))
# annot=True writes correlation coefficients in cells
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("sns.heatmap - Numerical Feature Correlation Matrix")
plt.show()
```

---

### 8. `sns.pairplot()`
* **Definition:** Draws pairwise scatter plot grid relationships across all numerical columns in a DataFrame.
* **Syntax:** `sns.pairplot(data=df, hue='cat_col')`

```python
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris")

# Pairplot creates a matrix grid of all numerical variables
g = sns.pairplot(iris, hue="species", palette="husl")
g.fig.suptitle("sns.pairplot - Iris Dataset Pairwise Feature Matrix", y=1.02)
plt.show()
```

---

## 18. Difference Table

### 1. Matplotlib vs Seaborn
| Comparison Metric | Matplotlib | Seaborn |
| :--- | :--- | :--- |
| **Level** | Low-Level graphic foundation | High-Level statistical wrapper |
| **Data Format** | Prefers NumPy arrays or lists | Prefers Pandas DataFrames |
| **Syntax** | Requires manual customization lines | Concise high-level functions |
| **Themes** | Needs explicit manual colors | Professional built-in themes |
| **Regression Lines** | Must calculate using `np.polyfit()` | Built-in via `sns.regplot()` / `sns.lmplot()` |

---

### 2. Bar Chart vs Histogram
| Attribute | Bar Chart | Histogram |
| :--- | :--- | :--- |
| **Data Type** | Categorical data (Discrete) | Continuous numerical data |
| **Bar Spacing** | Gaps between bars indicate distinct categories | Touch each other (no gaps) to represent continuous ranges |
| **X-Axis** | Category names (e.g., Python, Java) | Numeric Bins/Ranges (e.g., 0-10, 10-20) |
| **Reordering** | Bars can be reordered freely | Bin order is strictly numeric & fixed |

---

### 3. Scatter Plot vs Line Plot
| Attribute | Scatter Plot | Line Plot |
| :--- | :--- | :--- |
| **Data Continuity** | Independent discrete data points | Sequential/Continuous data over time |
| **Connection** | Points are unconnected | Points are connected with continuous lines |
| **Primary Goal** | Finds correlation / pattern between 2 variables | Tracks trends across chronological order |

---

### 4. Pie Chart vs Bar Chart
| Attribute | Pie Chart | Bar Chart |
| :--- | :--- | :--- |
| **Whole Requirement** | Must sum to 100% (Parts of a whole) | Independent values (Does not need to sum to 100%) |
| **Comparison Ease** | Hard for humans to accurately compare angles | Easy to compare relative bar lengths |
| **Category Limit** | Performs poorly above 5 slices | Handles dozens of categories easily |

---

## 19. Common Errors

Here are frequent mistakes beginner students make when writing Matplotlib & Seaborn code:

### 1. Forgetting `plt.show()`
* **Symptom:** In standalone Python scripts, code executes without errors but no visual window appears.
* **Fix:** Always include `plt.show()` at the end of chart code blocks.

### 2. Calling `plt.savefig()` AFTER `plt.show()`
* **Symptom:** Saved output PNG image is completely blank/white.
* **Explanation:** `plt.show()` clears the internal canvas state. Calling `savefig()` after `show()` saves an empty canvas.
* **Fix:** Always call `plt.savefig()` BEFORE `plt.show()`.

```python
# ❌ INCORRECT (Saves empty image)
plt.plot(x, y)
plt.show()
plt.savefig("my_plot.png")

# ✅ CORRECT
plt.plot(x, y)
plt.savefig("my_plot.png")
plt.show()
```

### 3. Mismatched Array Lengths
* **Error:** `ValueError: x and y must have same first dimension`
* **Fix:** Ensure `len(x) == len(y)` before calling `plt.plot()` or `plt.scatter()`.

### 4. Overlapping X-Axis Labels
* **Symptom:** Category labels collide and become unreadable.
* **Fix:** Add `plt.xticks(rotation=45)` or switch to horizontal bar chart (`plt.barh()`).

---

## 20. Best Practices

1. **Keep it Simple:** Avoid unnecessary 3D effects, excessive gridlines, or clutter that distracts from data.
2. **Label Everything:** Always supply an informative Title, X-axis label, and Y-axis label with units.
3. **Choose Appropriate Colors:** Use high-contrast color palettes. Use colorblind-friendly palettes (e.g., `viridis`, `cividis`).
4. **Use Legends Wisely:** Place legends outside the plot area if they overlap with data points (`plt.legend(bbox_to_anchor=(1.05, 1))`).
5. **Start Y-Axis at Zero:** Truncating the Y-axis can distort visual proportions and mislead viewers.

---

## 21. Practice Questions

Use these 20 classroom questions to test student comprehension:

1. What is the standard industry alias used when importing `matplotlib.pyplot`?
2. Which function is used to create a canvas window with custom dimensions in Matplotlib?
3. How do you save a plot to a PNG image file with 300 DPI resolution?
4. What parameter in `plt.plot()` changes the marker shape to squares?
5. Write code to rotate X-axis tick labels by 90 degrees.
6. What is the difference between `plt.bar()` and `plt.barh()`?
7. Explain the function of the `alpha` parameter in scatter plots.
8. How does `bins=15` impact a histogram plot?
9. In a pie chart, what does `autopct='%1.2f%%'` do?
10. What parameter is used to pull a slice out of a pie chart?
11. List the 5 metrics represented in a standard Box Plot summary.
12. How do you display a grid with dotted lines in Matplotlib?
13. What is the difference between `plt.subplot(2, 1, 1)` and `plt.subplot(1, 2, 1)`?
14. Which Seaborn function is best suited for drawing correlation heatmaps?
15. How do you overlay a Kernel Density Estimate (KDE) curve on a `sns.histplot()`?
16. What is the purpose of the `hue` parameter in Seaborn plots?
17. Name 3 built-in datasets provided by Seaborn.
18. Which Seaborn function creates a matrix grid of scatter plots for all numeric features?
19. Why does `plt.savefig()` save a blank image if called after `plt.show()`?
20. Why are histograms preferred over bar charts for continuous data like heights or test scores?

---

## 22. Assignments

### Assignment 1: Weekly Temperature Tracking
* **Task:** Create a line plot tracking daily temperatures (Monday to Sunday) for two cities (New York vs Miami).
* **Requirements:** Custom colors, distinct markers, grid lines, legend, and appropriate titles.

### Assignment 2: Company Department Headcount
* **Task:** Render a horizontal bar chart showing headcount across 6 departments (Engineering, HR, Sales, Marketing, Finance, Support).
* **Requirements:** Sort bars in ascending order, set edge color to black, and annotate count values on bars.

### Assignment 3: Student Study Hours vs Exam Score
* **Task:** Generate a scatter plot comparing study hours (X) with final scores (Y) for 30 students.
* **Requirements:** Color dots based on scores using `cmap='coolwarm'`, add a linear trendline.

### Assignment 4: Employee Salary Distribution
* **Task:** Plot a histogram of 100 employee salaries ranging from $30K to $150K.
* **Requirements:** Use 12 bins, black edges, and overlay a Seaborn KDE distribution curve.

### Assignment 5: Household Expense Budget Share
* **Task:** Create a styled Donut Chart showing monthly budget breakdown (Rent, Food, Utilities, Savings, Entertainment).
* **Requirements:** Show percentages to 1 decimal place, explode the "Savings" slice.

### Assignment 6: Multi-Department Boxplot Comparison
* **Task:** Compare performance score distributions across 4 regional branches using `sns.boxplot()`.
* **Requirements:** Highlight outliers with red square markers.

### Assignment 7: Subplot Dashboard
* **Task:** Create a 2x2 grid dashboard showing Line, Bar, Scatter, and Histogram plots using sample sales data.
* **Requirements:** Use `plt.tight_layout()` to prevent overlapping text.

### Assignment 8: Seaborn Restaurant Tips Analysis
* **Task:** Using Seaborn's `tips` dataset, plot a `sns.barplot()` showing average tip amount per day broken down by `smoker` status.

### Assignment 9: Feature Correlation Matrix Heatmap
* **Task:** Load Seaborn's `iris` dataset, compute feature correlations, and render a annotated heatmap with `cmap='YlGnBu'`.

### Assignment 10: Complete EDA Visualization Script
* **Task:** Take any custom CSV file, perform univariate analysis (Histograms/Boxplots) and bivariate analysis (Scatter/Barplots), and save all figures to a dedicated output folder.

---

## 23. Interview Questions

### Q1: What is the main difference between Matplotlib and Seaborn?
**Answer:** Matplotlib is a low-level graphics engine offering fine-grained pixel control, while Seaborn is a high-level statistical library built on top of Matplotlib that integrates with Pandas DataFrames and provides sophisticated default visual styles.

### Q2: How do you handle overlapping tick labels on the X-axis?
**Answer:** By rotating the labels using `plt.xticks(rotation=45)`, by scaling the figure width (`figsize`), or by rendering a horizontal chart (`plt.barh()`).

### Q3: What is the purpose of `plt.tight_layout()`?
**Answer:** `plt.tight_layout()` automatically adjusts subplot padding and margins to prevent overlapping titles, tick labels, and axis labels.

### Q4: Explain the 5-number summary in a Box Plot.
**Answer:** Minimum ($Q1 - 1.5 \times IQR$), First Quartile ($Q1$), Median ($Q2$), Third Quartile ($Q3$), and Maximum ($Q3 + 1.5 \times IQR$). Points outside this range are classified as outliers.

### Q5: How do you add a categorical grouping dimension to a Seaborn scatter plot?
**Answer:** By passing the categorical column name to the `hue` parameter (`sns.scatterplot(data=df, x='X', y='Y', hue='Category')`).

### Q6: What does the `explode` parameter do in `plt.pie()`?
**Answer:** It separates slice segments outward from the center of the pie chart by a specified fractional radius offset.

### Q7: Why is `plt.savefig()` executed before `plt.show()`?
**Answer:** `plt.show()` clears the figure canvas buffer upon rendering. If `plt.savefig()` is called afterwards, it captures an empty canvas resulting in a blank image file.

### Q8: What Seaborn plot is used to view all numeric variable pairwise relationships at once?
**Answer:** `sns.pairplot()`.

### Q9: How do you overlay a KDE line on a Seaborn histogram?
**Answer:** By setting the parameter `kde=True` in `sns.histplot()`.

### Q10: What is the difference between a Bar Chart and a Histogram?
**Answer:** A Bar chart visualizes discrete categorical comparisons with gaps between bars, whereas a Histogram visualizes continuous numerical ranges with contiguous bars.

### Q11: How do you change figure resolution in Matplotlib?
**Answer:** By setting the `dpi` (dots per inch) parameter in `plt.figure(dpi=300)` or `plt.savefig(dpi=300)`.

### Q12: How do you draw horizontal line reference thresholds on a plot?
**Answer:** Using `plt.axhline(y=value, color='r', linestyle='--')`.

### Q13: What parameter makes a Box Plot notched?
**Answer:** `notch=True` in `plt.boxplot()` or `sns.boxplot()`.

### Q14: How do you write floating-point percentages inside pie chart slices?
**Answer:** By passing a format specifier to `autopct`, such as `autopct='%1.1f%%'`.

### Q15: What Seaborn chart is used to count categorical occurrences?
**Answer:** `sns.countplot()`.

### Q16: How do you change the aesthetic theme across all Seaborn plots globally?
**Answer:** Using `sns.set_theme(style='darkgrid')` or `sns.set_style('whitegrid')`.

### Q17: What function is used to create object-oriented subplots?
**Answer:** `fig, axes = plt.subplots(nrows, ncols)`.

### Q18: How do you adjust line transparency in Matplotlib?
**Answer:** By passing a floating value between `0.0` (transparent) and `1.0` (opaque) to the `alpha` parameter.

### Q19: Which plot type is best suited for detecting correlation between two continuous variables?
**Answer:** Scatter Plot (`plt.scatter()` / `sns.scatterplot()`).

### Q20: How do you plot custom colors using Hex color codes in Matplotlib?
**Answer:** Pass hex strings directly into color parameters: `color='#1f77b4'`.

### Q21: What parameter in `sns.heatmap()` displays numerical values inside cells?
**Answer:** `annot=True`.

### Q22: What does `cmap` stand for and how is it used?
**Answer:** `cmap` stands for Color Map. It maps numerical data values to color gradients (e.g., `'viridis'`, `'coolwarm'`, `'plasma'`).

### Q23: How do you change legend location in Matplotlib?
**Answer:** Using `plt.legend(loc='upper right')` or using explicit coordinates via `bbox_to_anchor`.

### Q24: What is the purpose of `plt.clf()` and `plt.close()`?
**Answer:** `plt.clf()` clears the current active figure state, while `plt.close()` closes the figure window memory buffer entirely.

### Q25: How do you show frequency counts on top of bars in Matplotlib?
**Answer:** By iterating over returned bar rectangle objects and calling `plt.text()` for each bar height coordinate.

### Q26: What Seaborn chart combines box plot and KDE density distribution shape?
**Answer:** Violin Plot (`sns.violinplot()`).

### Q27: How do you set shared X or Y axes across subplots?
**Answer:** By passing `sharex=True` or `sharey=True` in `plt.subplots()`.

### Q28: How do you limit axis range boundaries?
**Answer:** Using `plt.xlim(min_val, max_val)` and `plt.ylim(min_val, max_val)`.

### Q29: What parameter in `plt.hist()` normalizes frequency counts to sum to 1?
**Answer:** `density=True`.

### Q30: How do you display a regression fit line in Seaborn?
**Answer:** Using `sns.regplot()` or `sns.lmplot()`.

---
