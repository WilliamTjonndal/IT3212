# IT3212 Assignment 3: Basic modelling

## Table of Contents

- [1. Develop a problem statement (real world and machine learning)](#1-problem-statement)
  - [a. This is one of the most important skills that a Machine Learning Engineer Scientist should have. Select a dataset and frame a machine learning problem and then connect this machine learning problem to the real world scenario. ](#problem-statement-section-1)
- [2. Implement the preprocessing and justify the preprocessing steps](#2-preprocessing)
- [3. Extract features and justify the methods used](#3-extract-features)
- [4. Select features and justify the methods used](#4-select-feactures)
- [5. Implement five out of the following algorithms and justify the choice](#5-implement-algorithms)
  - [a. Logistic regression](#implement-algorithms-section-1)
  - [b. Additive model](#implement-algorithms-section-2)
  - [c. Random forest](#implement-algorithms-section-3)
  - [d. SVM with kernels](#implement-algorithms-section-4)
  - [e. Neural Network](#implement-algorithms-section-5)
- [6. Compare the performance of the five algorithms with respect to your problem, explain the results](#6-compare-performance)
- [7. Implement boosting and bagging with your choice of base models and explain all the steps](#7-boosting-bagging)
- [8. Implement one instance of transfer learning (find a related bigger dataset online) and explain all the steps](#8-transfer-learning)
  - [a. Explain the bigger dataset with visualization and summary statistics.](#transfer-learning-section-1)

- [9. Compare the performance of the algorithms (basic VS boosting VS bagging VS transfer) with respect to your machine learning problem and explain the results](#7-compare-performance)


<div style="page-break-after: always;"></div>

## <a id="1-problem-statement"></a> 1. Develop a problem statement (real world and machine learning)

### <a id="#problem-statement-section-1"></a> a. This is one of the most important skills that a Machine Learning Engineer/Scientist should have. Select a dataset and frame a machine learning problem and then connect this machine learning problem to the real world scenario.

**Real World Problem** \
As the education sector becomes more data-driven, collected data can unlock substansial value. Universities want to reduce first-year dropout and capture students who are likely to still be enrolled beyond the normal time to degree, so insititutions can allocate extra resources proactively and help students gets back on track. This improves student success and workforce readiness, strengthens institutional outcomes, and generates insights useful for policymakers.

**Machine Learning Problem** \
With this in mind, we selected the Student Graduation dataset, which records students across multiple undergraduate programs and includes socio-economic factors, prior academic background, and performance at the end of the first and second semesters. Our goal is to train machine learning models that predict three outcomes: dropout, extended enrollment beyond the normal time, or successful completion of the first year. These predictions directly support the real-world problem by enabling early, targeted interventions for students at risk.


<div style="page-break-after: always;"></div>
## <a id="2-preprocessing"></a> 2. Implement the preprocessing and justify the preprocessing steps

<div style="page-break-after: always;"></div>

## <a id="3-extract-features"></a> 3. Extract features and justify the methods used

<div style="page-break-after: always;"></div>

## <a id="4-select-feactures"></a> 4. Select features and justify the methods used

<div style="page-break-after: always;"></div>

## <a id="5-implement-algorithms"></a> 5. Implement five out of the following algorithms and justify the choice

### <a id="implement-algorithms-section-1"></a> a. Logistic regression 

**How it works**

Multinomial logistic regression models the log odds of each class as a linear function of the inputs and uses a softmax layer to output class probabilities.

**Why we chose it**

It is a strong baseline for multiclass classification, works well with our one-hot encoded categorical features, and is easy to interpret through its coefficients. This makes it easy to evaluate the reliability of the model by confirming that it captures reasonable relationships between social-economic factors and the student's academic performance. A known limitation is the linearity assumption, which can miss non-linear socio-economic patterns.

<div style="page-break-after: always;"></div>

### <a id="implement-algorithms-section-2"></a> b. Additive model

**How it works**

A generalized additive model (GAM) represents the log odds as a sum of smooth functions of each feature, often via splines, which captures nonlinear shapes without manual feature engineering.

**Why we chose it**

Variables such as age at enrollment, admission grade, and approved units often have curved and thresholded effects. GAMs model these patterns directly while remaining interpretable, which improved our classification.

<div style="page-break-after: always;"></div>

### <a id="implement-algorithms-section-3"></a> c. Random forest

**How it works**

A random forest builds many decision trees on bootstrap samples while randomly selecting subsets of features at each split. The final prediction is the majority vote across trees.

**Why we chose it**

It usually delivers higher accuracy than a single tree and handles many attributes well, including our one-hot encoded features and mixed numeric inputs. Although ensembles can be computationally heavier, our dataset is small enough that training is efficient, and we also gain useful feature importance signals.

<div style="page-break-after: always;"></div>

### <a id="implement-algorithms-section-4"></a> d. SVM with kernels

**How it works**

A support vector machine (SVM) finds a maximum margin boundary. With kernels such as the radial basis function it implicitly maps data to a higher dimensional space to separate complex patterns, relying on support vectors at decision boundaries.

**Why we chose it**

It performs well in high dimensional spaces created by one hot encoding and often gives strong accuracy with good regularization. Prediction is fast compared to Naive Bayes and it use less memory since it only uses a subset of the training points in the decision phase. Training can be slow on very large data, but our dataset size makes it a good fit.

<div style="page-break-after: always;"></div>

### <a id="implement-algorithms-section-5"></a> e. Neural networks

**How it works**

A feed forward neural network stacks linear layers with nonlinear activations and learns parameters by backpropagation. For multi class outputs it ends with a softmax layer to produce probabilities.

**Why we chose it**

It can learn complex interactions among demographic, financial, and academic features that simpler linear models may miss. With proper scaling, regularization, and early stopping, it complements the other methods by offering a representation learning approach that can raise predictive performance on structured data.

## <a id="6-compare-performance"></a> 6. Compare the performance of the five algorithms with respect to your problem, explain the results

<div style="page-break-after: always;"></div>

## <a id="7-boosting-bagging"></a> 7. Implement boosting and bagging with your choice of base models and explain all the steps

We have implemented four ensemble learning methods: **Bagging with Decision Trees**, **Bagging with SVM**, **AdaBoost**, and **XGBoost**, using a slightly ***modified pipeline*** than the one used previously.

The **Bagging (Decision Tree)** model trains multiple `decision trees` on different bootstrap samples, where each `tree` learns independently using random subsets of both observations and features.\
This parallel training reduces variance, stabilizes the predictions, and relies on majority voting to produce the final class output.

The **Bagging (SVM)** model follows the same sampling strategy but uses `Support Vector Machines` as base learners, allowing each `SVM` to learn slightly different decision boundaries.\
These independent `SVM` then vote to determine the ensemble prediction, improving robustness on noisy datasets.

The **AdaBoost model**, using `decision trees` as weak learners, builds its ensemble sequentially by increasing the weight of misclassified samples after each iteration.\
This causes later learners to focus on difficult cases, improving bias reduction through weighted voting.

Finally, the **XGBoost model** constructs `boosted trees` using gradient-based optimization, where each new `tree` corrects residual errors from earlier ones while applying regularization, subsampling, and column sampling to control overfitting and enhance generalization.

The ***modified pipeline*** first prepares the dataset by applying a `SimpleImputer` with a median strategy to handle missing values and by structuring all models within an integrated preprocessing–model pipeline.\
**Hyperparameter tuning** is performed using `GridSearchCV`, evaluating combinations such as learning rate, maximum depth, number of estimators, and sampling ratios across stratified cross-validation folds to ensure robust model comparison.\
Each ensemble model is then refitted using the best-found parameters and evaluated using the test set through accuracy and balanced accuracy metrics.

<div style="display: flex;">
  <figure style="text-align: center; margin: 25 5px 25 0;">
    <img src="img/confusion_matrix_bagging_dt.png" width="400"/>
    <figcaption><em>Figure 2.a: Confusion matrix (Decision Trees)</em></figcaption>
  </figure>
  <figure style="text-align: center; margin: 25 5px 25 0;">
    <img src="img/confusion_matrix_bagging_svm.png" width="400"/>
    <figcaption><em>Figure 2.b: Confusion matrix (Support Vector Machines)</em></figcaption>
  </figure>

  <figure style="text-align: center; margin: 25 5px 25 0;">
    <img src="img/confusion_matrix_boosting_adaboost.png" width="400"/>
    <figcaption><em>Figure 2.c: Confusion matrix (AdaBoost)</em></figcaption>
  </figure>
  <figure style="text-align: center; margin: 25 5px 25 0;">
    <img src="img/confusion_matrix_boosting_xgb.png" width="400"/>
    <figcaption><em>Figure 2.d: Confusion matrix (XGBoost)</em></figcaption>
  </figure>
</div>

As seen in the confusion matrices aboves, we have improved our prediction rate for classes 1 and 2 (Dropout and Graduate respectively).\
However, the models are still biased towards class 2 (Graduate). In some cases, the models perfome worse when predicting class 1 (Enrolled).\
This happens because enrolled students share characteristics with both dropouts and graduates, making them a “middle-ground” class, which is difficult for our models.

<div style="display: flex;">
  <figure style="text-align: center; margin: 25 10px 25 0;">
    <img src="img/bagging_vs_boosting_accuracy.png" width="400"/>
    <figcaption><em>Figure 3a: Bagging vs Boosting accuracy</em></figcaption>
  </figure>
  <figure style="text-align: center; margin: 25 10px 25 0;">
    <img src="img/bagging_vs_boosting_balanced_accuracy.png" width="400"/>
    <figcaption><em>Figure 3b: Bagging vs Boosting balanced accuracy</em></figcaption>
  </figure>
  <figure style="text-align: center; margin: 25 10px 25 0;">
    <img src="img/bagging_vs_boosting_macro_precision.png" width="400"/>
    <figcaption><em>Figure 3c: Bagging vs Boosting macro precisison accuracy</em></figcaption>
  </figure>
</div>

By **Accuracy** we mean the percentage of all predictions that were correct.\
From ***Figure 3a***, we notice that **accuracy** ranges from 0.729 to 0.777 across models. Based on this mean, the best model, `Bagging DT` (Bagging with Decision Trees), correctly predicts ~78% of students.

However, accuracy can be misleading because the classes are unbalanced (many students are "graduates").\
To resolve this, we have have also used **Balanced Accuracy**. This is the mean recall across all classes, which gives each class equal weight.\
**Balanced accuracy** goes from 0.691 to 0.703 which is lower than raw accuracy.\
This indicates that the models predict class 2 (Graduate) well, but  struggle with class 1 (Enrolled). Once again, `Bagging DT` again performs best with 0.703.

Finally, **Macro precision** measures correctness per class, averaged equally. It goes from 0.68 up to 0.73. Once again, `Bagging DT` perfomed the best. This means that when the model predicts a class, it is correct roughly 73% of the time.\
`AdaBoost` performed the worse, because its weak learners cannot model the complexity of our imbalanced dataset, causing more misclassifications and therefore more false positives in multiple classes.

<p align="center">
<img src="img/roc_comparison.png" width="1200"/><br>
<em>Figure 4: ROC graphs for all models</em>
</p>

**ROC-AUC** measures how well the model separates classes. Our models go from 0.84 to 0.88 which is good.\
`XGBoost` and `Bagging DT` show the best class separation.

<u>To conclude:</u>
- All models do very well at detecting graduates.
- All models struggle with the Enrolled class.
- `Bagging DT` and `XGBoost` are the strongest overall.
- **ROC-AUC** shows the models have good separability.


<div style="page-break-after: always;"></div>

## <a id="8-transfer-learning"></a> 8. Implement one instance of transfer learning (find a related bigger dataset online) and explain all the steps

### <a id="transfer-learning-section-1"></a> a. Explain the bigger dataset with visualization and summary statistics.

<div style="page-break-after: always;"></div>

## <a id="7-compare-performance"></a> 9. Compare the performance of the algorithms (basic VS boosting VS bagging VS transfer) with respect to your machine learning problem and explain the results

<div style="page-break-after: always;"></div>
