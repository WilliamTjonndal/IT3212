# IT3212 Assignment 1: Data Preprocessing

<!-- Create a table of contents here) -->

## 1. Data Exploration

### a. Explore the dataset by displaying the first few rows, summary statistics, and data types of each column.

We have chosen the Stock market dataset.\

the dataset contains contains the following columns:

- ``Date``: The date the stock was traded (datetime)
- ``Open``: Price of the first stock that was traded on that date (float)
- ``High``: Highest price the stock was traded for on that date (float)
- ``Low``: Lowest price the stock was traded for on that date (float)
- ``Close``: Last price the stock was traded for on that date (float)
- ``Volume``: Number of traded stocks that (integer)
- ``OpenInt``: Open interest, number of stocks that are still open to be traded that date (integer)
- ``Symbol``: Stock symbol, abbreviation used to identify a stock (string)

<img src="img/first_few_rows.png" width="500"/>

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; **Figure 1**: First few rows of the dataset


### b. Identify missing values, outliers, and unique values in categorical columns.

```py
```


## 2. Data Cleaning

### a. Handling Missing Values

The dataset contains some empty files, which we obviously ignore.\
Some files are not properly formatted csv files. How we handle with such files is described belows.

### b. Choose appropriate methods to handle missing values (e.g., mean/median imputation for numerical data, mode imputation for categorical data, or deletion of rows/columns).

Note: The ``OpenInt`` column is always **0**, so we will ignore it completely.

### c. Justify your choices for handling missing data.

```py
```


## 3. Handling Outliers

### a. Detect outliers using methods such as the IQR method or Z-score.



### b. Decide whether to remove, cap, or transform the outliers. Justify your decisions.

```py
```


## 4. Data Transformation

### a. Encoding Categorical Data

#### i. Apply label encoding or one-hot encoding to transform categorical data into numerical form.

```py
```

#### ii. Justify your choice of encoding method.



### b. Feature Scaling

#### i. Apply feature scaling techniques such as normalization (Min-Max scaling) or standardization (Z-score normalization) to the dataset.



#### ii. Explain why feature scaling is necessary and how it impacts the model.

Feature scaling is important because raw features often have very different ranges, and this can cause models to give more weight to features with larger values. By scaling, we ensure that all features contribute equally, which improves fairness and accuracy.

## 5. Data Splitting

### a. Split the preprocessed dataset into training and testing sets. Typically, an 80-20 or 70-30 split is used.

```py
```

### b. Explain the importance of splitting the data and how it prevents overfitting.

Splitting the data allows the model to be trained on one set and evaluated on another, ensuring that performance is measured on unseen data. The training set adjusts model parameters, while the test set checks generalization. This prevents overfitting by forcing the model to learn patterns instead of memorizing the training data. A validation set is often used during training to tune hyperparameters and monitor performance.


## 6. Apply dimensionality reduction techniques such as Principal Component Analysis (PCA) and discuss how it affects the dataset.

```py
```