# Week 1 Data Science Cheat Sheet

## 1. Dataset Basics

| Term | Meaning |
|---|---|
| Row | One observation or one data point |
| Column | One feature or variable |
| Target | The value we want to predict |
| Feature | Input used for prediction |
| Numerical column | Contains numbers |
| Categorical column | Contains groups or labels |

## 2. EDA Checklist

1. Load dataset.
2. Check shape.
3. Check first few rows.
4. Check data types.
5. Check missing values.
6. Check duplicates.
7. Study target distribution.
8. Analyze numerical features.
9. Analyze categorical features.
10. Study relationships with the target.

## 3. Common Pandas Commands

```python
df.head()
df.info()
df.describe()
df.shape
df.columns
df.isnull().sum()
df.duplicated().sum()
df["column"].value_counts()
df.groupby("column")["target"].mean()
```

## 4. Feature Engineering

Feature engineering means creating better input columns from existing data.

Examples:

```python
df["family_size"] = df["sibsp"] + df["parch"] + 1
df["is_alone"] = (df["family_size"] == 1).astype(int)
df["age_group"] = pd.cut(df["age"], bins=[0, 12, 18, 35, 60, 100])
```

## 5. Missing Value Handling

```python
df["age"].fillna(df["age"].median(), inplace=True)
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)
```

Better approach in ML pipeline:

```python
SimpleImputer(strategy="median")
SimpleImputer(strategy="most_frequent")
```

## 6. Encoding

Categorical data must be converted into numbers.

```python
pd.get_dummies(df, drop_first=True)
```

Pipeline approach:

```python
OneHotEncoder(handle_unknown="ignore")
```

## 7. Scaling

Scaling brings numerical features to a similar range.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## 8. Vectors and Matrices

| Concept | Meaning in Data Science |
|---|---|
| Scalar | Single number |
| Vector | One row / one data point |
| Matrix | Full dataset |
| Dot product | Similarity or weighted sum |

```python
import numpy as np

vector = np.array([1, 2, 3])
matrix = np.array([[1, 2, 3], [4, 5, 6]])
dot = np.dot(vector, vector)
```

## 9. PCA

PCA reduces many features into fewer important components.

Three-line practical PCA:

```python
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

Use PCA for:

- visualization
- dimensionality reduction
- feature compression
- reducing correlated features

## 10. GitHub Portfolio Checklist

Before pushing:

- Notebook runs from top to bottom.
- Markdown explains every section.
- Code is clean.
- Plots have titles and labels.
- File names are professional.
- README explains the project.
- requirements.txt is included.

## 11. Classical ML Libraries

Install these:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels xgboost lightgbm
```

## 12. Models to Learn Next

- Linear Regression
- Logistic Regression
- KNN
- Decision Tree
- Random Forest
- Naive Bayes
- Support Vector Machine
- XGBoost