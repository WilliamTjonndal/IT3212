# IT3212 Assignment 1: Data Preprocessing

### Group 25

## Table of Contents

- [IT3212 Assignment 1: Data Preprocessing](#it3212-assignment-1-data-preprocessing)
    - [Group 25](#group-25)
  - [Table of Contents](#table-of-contents)
  - [1. Data Exploration](#1-data-exploration)
    - [a. Explore the dataset by displaying the first few rows, summary statistics, and data types of each column.](#a-explore-the-dataset-by-displaying-the-first-few-rows-summary-statistics-and-data-types-of-each-column)
    - [b. Identify missing values, outliers, and unique values in categorical columns.](#b-identify-missing-values-outliers-and-unique-values-in-categorical-columns)
  - [2. Data Cleaning](#2-data-cleaning)
    - [a. Handling Missing Values](#a-handling-missing-values)
    - [b. Choose appropriate methods to handle missing values (e.g., mean/median imputation for numerical data, mode imputation for categorical data, or deletion of rows/columns).](#b-choose-appropriate-methods-to-handle-missing-values-eg-meanmedian-imputation-for-numerical-data-mode-imputation-for-categorical-data-or-deletion-of-rowscolumns)
    - [c. Justify your choices for handling missing data.](#c-justify-your-choices-for-handling-missing-data)
  - [3. Handling Outliers](#3-handling-outliers)
    - [a. Detect outliers using methods such as the IQR method or Z-score.](#a-detect-outliers-using-methods-such-as-the-iqr-method-or-z-score)
    - [b. Decide whether to remove, cap, or transform the outliers. Justify your decisions.](#b-decide-whether-to-remove-cap-or-transform-the-outliers-justify-your-decisions)
  - [4. Data Transformation](#4-data-transformation)
    - [a. Encoding Categorical Data](#a-encoding-categorical-data)
      - [i. Apply label encoding or one-hot encoding to transform categorical data into numerical form.](#i-apply-label-encoding-or-one-hot-encoding-to-transform-categorical-data-into-numerical-form)
      - [ii. Justify your choice of encoding method.](#ii-justify-your-choice-of-encoding-method)
    - [b. Feature Scaling](#b-feature-scaling)
      - [i. Apply feature scaling techniques such as normalization (Min-Max scaling) or standardization (Z-score normalization) to the dataset.](#i-apply-feature-scaling-techniques-such-as-normalization-min-max-scaling-or-standardization-z-score-normalization-to-the-dataset)
      - [ii. Explain why feature scaling is necessary and how it impacts the model.](#ii-explain-why-feature-scaling-is-necessary-and-how-it-impacts-the-model)
  - [5. Data Splitting](#5-data-splitting)
    - [a. Split the preprocessed dataset into training and testing sets. Typically, an 80-20 or 70-30 split is used.](#a-split-the-preprocessed-dataset-into-training-and-testing-sets-typically-an-80-20-or-70-30-split-is-used)
    - [b. Explain the importance of splitting the data and how it prevents overfitting.](#b-explain-the-importance-of-splitting-the-data-and-how-it-prevents-overfitting)
  - [6. Apply dimensionality reduction techniques such as Principal Component Analysis (PCA) and discuss how it affects the dataset.](#6-apply-dimensionality-reduction-techniques-such-as-principal-component-analysis-pca-and-discuss-how-it-affects-the-dataset)

## 1. Data Exploration

### a. Explore the dataset by displaying the first few rows, summary statistics, and data types of each column.

We have chosen the Stock market dataset. The dataset contains contains the following columns:

- `Date`: The date the stock was traded (datetime)
- `Open`: Price of the first stock that was traded on that date (float)
- `High`: Highest price of the stock was traded on that date (float)
- `Low`: Lowest price of the stock that was traded on that date (float)
- `Close`: Last price of the stock that was traded on that date (float)
- `Volume`: Number of traded stocks on that date (integer)
- `OpenInt`: Open contract, number of stocks that are still open to be traded on that date (integer)
- `Symbol`: Stock symbol, abbreviation used to identify a stock (string)

<p align="center">
  <img src="img/datatypes.png" width="200"/><br>
  <em>Figure 1: Data types for each column</em>
</p>

<p align="center">
  <img src="img/first_few_rows.png" width="600"/><br>
  <em>Figure 2: First few rows of the dataset</em>
</p>

Below are a few figures describing some summary statistics of the dataset.

<p align="center">
  <img src="img/initial_statistics.png" width="600"/><br>
  <em>Figure 3: Initial statistics of the dataset</em>
</p>

The initial summary statistics also revealed several sudden significant changes in mean and median price.

<p align="center">
  <img src="img/open_over_time_with_count.png" width="500"/><br>
  <em> Figure 4: Open price over time</em>
</p>

<p align="center">
  <img src="img/high_over_time_with_count.png" width="500"/><br>
  <em>Figure 5: High price over time</em>
</p>

<p align="center">
  <img src="img/low_over_time_with_count.png" width="500"/><br>
  <em>Figure 6: Low price over time</em>
</p>

<p align="center">
  <img src="img/close_over_time_with_count.png" width="500"/><br>
  <em>Figure 7: Close price over time</em>
</p>

<p align="center">
  <img src="img/volume_over_time_with_count.png" width="500"/><br>
  <em>Figure 8: Volume over time</em>
</p>

### b. Identify missing values, outliers, and unique values in categorical columns.

When comparing the price trends with the number of recorded stock entries, we found that these drops were not caused by actual market movements but rather by missing data during those periods.

In the initial analysis, we observed that the open, high, low, and close prices of the stocks were relatively similar in terms of their mean and average values (see Figures 4–8).

We also observed a significant shift in stock prices around 1970 and 2005. As illustrated in Figure 9, this aligns with substantial increases in the number of recorded stocks.\
The correlation between the surge in available stock data (Visualized by the dotted green count line) and the change in price statistics suggests that the shift was primarily driven by the expansion of the dataset rather than by underlying market dynamics.

<p align="center">
  <img src="img/data_over_time.png" width="500"/><br>
  <em>Figure 9: Number of stock data points over time</em>
</p>

Boxplots for each column (Figure 10) reveal many high outliers, though they do not fully explain their causes.\
It is also important to note that market trends vary, and sharp drops do not always indicate unrealistic prices but can reflect real economic events, such as the 2008 financial crisis observed in Figures 4–8.

<p align="center">
  <img src="img/boxplots.png" width="500"/><br>
  <em>Figure 10: Boxplots for each column</em>
</p>

The unique values in categorial columns are the stock symbols, i.e., the identifiers for the different companies. See figure 11.\
The OpenInt column has the value 0 for every row.

<p align="center">
  <img src="img/unique_columns.png" width="200"/><br>
  <em>Figure 11: Unique values for each column</em>
</p>

In total, stock data was collected for 7195 companies. Of these, 32 files were empty and therefore unusable.\
For the retrieved companies, the collected rows (date with given stock prices for a company) contained no missing data as shown in figure 12, but figures 4-8 suggest that there might be some missing days of stock data.

<p align="center">
  <img src="img/missing_values.png" width="200"/><br>
  <em>Figure 12: Missing values for each column</em>
</p>

The dataset also contains some cells with negative values, this will be treated as invalid data as the price of a stock can't be below 0.

## 2. Data Cleaning

### a. Handling Missing Values

Below are the same figures as in Task 1, but now with forward-filling applied to the dataset.

<p align="center">
  <img src="img/open_over_time_with_count_after_filled.png" width="500"/><br>
  <em>Figure 13: Open price over time after forward-filling</em>
</p>

<p align="center">
  <img src="img/high_over_time_with_count_after_filled.png" width="500"/><br>
  <em>Figure 14: High price over time after forward-filling</em>
</p>

<p align="center">
  <img src="img/low_over_time_with_count_after_filled.png" width="500"/><br>
  <em>Figure 15: Low price over time after forward-filling</em>
</p>

<p align="center">
  <img src="img/close_over_time_with_count_after_filled.png" width="500"/><br>
  <em>Figure 16: Close price over time after forward-filling</em>
</p>

<p align="center">
  <img src="img/volume_over_time_with_count_after_filled.png" width="500"/><br>
  <em>Figure 17: Volume over time after forward-filling</em>
</p

After handling missing values, the fluctuations seen in Figures 4–8 are resolved, leaving only the major shifts around 1970 and 2005, which correlate with the previously discussed increases in available stock data.

<p align="center">
  <img src="img/statistics_after_filled.png" width="500"/><br>
  <em>Figure 18: Statistics of the dataset after forward-filling</em>
</p>

Despite forward-filling resolving the price fluctuations, outliers remain visible (Figure 19).

<p align="center">
  <img src="img/boxplots.png" width="500"/><br>
  <em>Figure 19: Boxplots for each column after forward-filling</em>
</p>

### b. Choose appropriate methods to handle missing values (e.g., mean/median imputation for numerical data, mode imputation for categorical data, or deletion of rows/columns).

As stated above, we used forward-fill to handle missing values.\
Filling in with average values would be wrong since the market could vary, so it is reasonable to fill in with the previous stock data.\
If there are missing stock data on Monday, we forward-fill with data from Friday.

We also decided to exclude weekends and market holidays.

The OpenInt column was completely dropped.

We ignore the 32 companies with empty stock data, dropped the open interest column, and apply forward-fill for missing values in price columns.

### c. Justify your choices for handling missing data.

The 32 companies represent a very small share of the dataset. We chose to ignore these companies to avoid creating stock data with nothing to base it on.\
The open interest column is excluded since it contains only zeros for all entries (see Figures 2 and 3), indicating no reported open contracts and adding no useful information for prediction while introducing unnecessary computation overhead.\
Forward-fill is used because missing prices are more likely close to the previous day's data rather than to mean or median price.\
We preferred filling with the previous day's data instead of the next day's data as otherwise that would fill our dataset using the knowledge of future prices.

## 3. Handling Outliers

### a. Detect outliers using methods such as the IQR method or Z-score.

We have decided to use Z-score to detect outliers.\
We use a per-stock rolling Z-score with a window of 1 month, filling all cells with a Z-score above 3 with the previous day's value for the same stock.

We also do this with all cells where the value is negative, as neither price nor volume can be negative.

Below are same figures as before, but with Z-score outliers removed. This doesnt have much of an effect on the graphs as we didn't find many outliers, around 5000 in each column in total.

<p align="center">
  <img src="img/open_zscore.png" width="600"/><br>
  <em>Figure 20: Open price over time after removing outliers</em>
</p>

<p align="center">
  <img src="img/high_zscore.png" width="600"/><br>
  <em>Figure 21: High price over time after removing outliers</em>
</p>

<p align="center">
  <img src="img/low_zscore.png" width="600"/><br>
  <em>Figure 22: Low price over time after removing outliers</em>
</p>

<p align="center">
  <img src="img/close_zscore.png" width="600"/><br>
  <em>Figure 23: Close price over time after removing outliers</em>
</p>

<p align="center">
  <img src="img/volume_zscore.png" width="600"/><br>
  <em>Figure 24: Volume over time after removing outliers</em>
</p>

### b. Decide whether to remove, cap, or transform the outliers. Justify your decisions.

All fields where we detected outliers were filled with the previous day's data.\
We have chosen to do this to be consistent with our method of filling cells, as we do the same with missing rows.\
Z-score is chosen to remove outliers, as it seems to be most used method for detecting outliers in stock data. Our group also has more experience using Z-score, so we're more familiar with how the method works, meaning we understand our new dataset better.\
Finally, rolling Z-score handles temporal fluctuations more effectively than global Z-score, which may flag small jumps after a normal price increase as an outlier.

We know Z-score can remove values for very volatile stocks, but since we only detected around 5000 outliers per column, we have decided that it's okay to get rid of these data points since our dataset is so large.

## 4. Data Transformation

### a. Encoding Categorical Data

#### i. Apply label encoding or one-hot encoding to transform categorical data into numerical form.

We have chosen to apply label encoding.\
Below are some summary statistics of the dataset after encoding the ``Symbol`` column.

<p align="center">
  <img src="img/statistics_after_label_encoding.png" width="600"/><br>
  <em>Figure 25: First few rows of the dataset after label encoding</em>
</p>

<p align="center">
  <img src="img/number_records_symbol.png" width="500"/><br>
  <em>Figure 26: Number of entries per stock symbol after label encoding</em>
</p>

<p align="center">
  <img src="img/avg_close_price_symbols.png" width="500"/><br>
  <em>Figure 27: Average close price for 20 stock symbols after label encoding</em>
</p>

<p align="center">
  <img src="img/avg_close_price_time_symbols.png" width="500"/><br>
  <em>Figure 28: Close price over time for stock symbol 0 after label encoding</em>
</p>

We chose to label encode the symbol column, setting an integer to replace every unique value of symbol in the dataset.\
We applied label encoding to the symbol column. Since encoding starts from 0, the count in Figure 19 is 7162, despite there being 7163 unique companies.

#### ii. Justify your choice of encoding method.

With 7,162 unique companies, one-hot encoding would introduce 7,162 extra features, making the model unnecessarily complex. Label encoding is thus more suitable in this case, as it assigns each unique company a numeric value, reducing dimensionality while preserving category distinction.

### b. Feature Scaling

#### i. Apply feature scaling techniques such as normalization (Min-Max scaling) or standardization (Z-score normalization) to the dataset.

We decided to use Min-Max normalization to scale the dataset.\
The scaling was only used on the columns that have ordinal data, meaning values have a meaningful order.\
These columns are Open price, High price, Low price, Close price, and Volume.

Min-Max was chosen instead of other scaling techniques, as the result of Min-Max scaling gives an intuitive understanding of the output.\
What you can tell from the result of Min-Max scaling is what percentage from the minimum value to the maximum value a give datapoint is.\
This is preferred over Z-score normalization, as we felt we don't have an intuitive understanding of it in the context of stock data.



<p align="center">
  <img src="img/after_min_max.png" width="500"/><br>
  <em>Figure 29: First few rows of the dataset after Min-Max scaling</em>
</p>


<p align="center">
  <img src="img/open_scaling.png" width="500"/><br>
  <em>Figure 30: Open price before and after Min-Max scaling</em>
</p>

<p align="center">
  <img src="img/high_scaling.png" width="500"/><br>
  <em>Figure 31: High price before and after Min-Max scaling</em>
</p>

<p align="center">
  <img src="img/low_scaling.png" width="500"/><br>
  <em>Figure 32: Low price before and after Min-Max scaling</em>
</p>

<p align="center">
  <img src="img/close_scaling.png" width="500"/><br>
  <em>Figure 33: Close price before and after Min-Max scaling</em>
</p>

<p align="center">
  <img src="img/volume_scaling.png" width="500"/><br>
  <em>Figure 34: Volume before and after Min-Max scaling</em>
</p>

We can see that after scaling we get a left-skewed distribution.\
This means that the stock prices are densely packed at the lower end of their own range.

#### ii. Explain why feature scaling is necessary and how it impacts the model.

Feature scaling is important because raw features often have very different ranges, and this can cause models to give more weight to features with larger values.\
By scaling, we ensure that all features contribute equally, which improves fairness and accuracy.

## 5. Data Splitting

### a. Split the preprocessed dataset into training and testing sets. Typically, an 80-20 or 70-30 split is used.

We split the dataset into a training and testing dataset using an 80-20 split.\
Instead of splitting the dataset randomly, we choose to use the chronologically last 20% of the dataset as testing data.\
This avoids data leakage, as the model isn't trained on events and values that aren't available at the time of the prediction.

```
Training set shape: (12634695, 7)
Testing set shape: (3163614, 7)
```

When training the model, we would prefer using chronological splits with rolling validation, as this method would mean most of the training data could be used for both validation and training without data leakage.

### b. Explain the importance of splitting the data and how it prevents overfitting.

Splitting the data allows the model to be trained on one set and evaluated on another, ensuring that performance is measured on unseen data.\
The training set adjusts model parameters, while the test set checks generalization. This prevents overfitting by forcing the model to learn patterns instead of memorizing the training data.\
A validation set is often used during training to tune hyperparameters and monitor performance.

## 6. Apply dimensionality reduction techniques such as Principal Component Analysis (PCA) and discuss how it affects the dataset.

<!-- the two paragraphs below seem to be AI generated, rewrite?-->

Applying Principal Component Analysis (PCA) to our stock dataset — which includes features such as Open, High, Low, Close, Volume, and OpenInt — helps reduce dimensionality by transforming the original correlated features into a smaller set of uncorrelated components. Since stock prices (Open, High, Low, Close) are often highly correlated, PCA captures most of their shared variance in the first few principal components.

This dimensionality reduction simplifies the dataset while retaining the majority of the information. As a result, the model becomes less complex, less prone to multicollinearity, and easier to visualize. However, interpretability decreases since the new principal components are linear combinations of the original features rather than directly meaningful attributes like “Open” or “Close” price.

<p align="center">
  <img src="img/PCA_Train_Head.png" width="250"/><br>
  <em>Figure 35: First few lines from the PCA train set</em>
</p>

<p align="center">
  <img src="img/PCA_Test_Head.png" width="250"/><br>
  <em>Figure 36: First few lines from the PCA test set</em>
</p>

<p align="center">
  <img src="img/PCA_vuisualized.png" width="500"/><br>
  <em>Figure 37: A random sample from the first two PCA components</em>
</p>

<!-- possibly human written?-->

Applying dimensionality reduction using PCA changes the dimensions of the dataset by replacing columns with a smaller number of new features. The effect is the reduction of dimensions, which can speed up model training and remove features that aren't valuable for training.

There are also problems with dimensionality reduction. One is that it makes the features in the data less intuitively understood, making it hard to understand the dataset and interpret predictions. Dimensionality reduction also loses some information, which might be valuable to training the model.