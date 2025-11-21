# IT3212 Assignment 3: Basic modelling

## Table of Contents

- [IT3212 Assignment 3: Basic modelling](#it3212-assignment-3-basic-modelling)
  - [Table of Contents](#table-of-contents)
  - [ 1. Develop a problem statement (real world and machine learning)](#-1-develop-a-problem-statement-real-world-and-machine-learning)
    - [ a. This is one of the most important skills that a Machine Learning Engineer/Scientist should have. Select a dataset and frame a machine learning problem and then connect this machine learning problem to the real world scenario.](#-a-this-is-one-of-the-most-important-skills-that-a-machine-learning-engineerscientist-should-have-select-a-dataset-and-frame-a-machine-learning-problem-and-then-connect-this-machine-learning-problem-to-the-real-world-scenario)
  - [ 2. Implement the preprocessing and justify the preprocessing steps](#-2-implement-the-preprocessing-and-justify-the-preprocessing-steps)
  - [ 3. Extract features and justify the methods used](#-3-extract-features-and-justify-the-methods-used)
  - [ 4. Select features and justify the methods used](#-4-select-features-and-justify-the-methods-used)
  - [ 5. Implement five out of the following algorithms and justify the choice](#-5-implement-five-out-of-the-following-algorithms-and-justify-the-choice)
    - [ a. Logistic regression](#-a-logistic-regression)
    - [ b. Decision trees](#-b-decision-trees)
    - [ c. Random forest](#-c-random-forest)
    - [ d. SVM with kernels](#-d-svm-with-kernels)
    - [ e. Neural networks](#-e-neural-networks)
  - [ 6. Compare the performance of the five algorithms with respect to your problem, explain the results](#-6-compare-the-performance-of-the-five-algorithms-with-respect-to-your-problem-explain-the-results)
  - [ 7. Implement boosting and bagging with your choice of base models and explain all the steps](#-7-implement-boosting-and-bagging-with-your-choice-of-base-models-and-explain-all-the-steps)
  - [ 8. Implement one instance of transfer learning (find a related bigger dataset online) and explain all the steps](#-8-implement-one-instance-of-transfer-learning-find-a-related-bigger-dataset-online-and-explain-all-the-steps)
    - [ a. Explain the bigger dataset with visualization and summary statistics.](#-a-explain-the-bigger-dataset-with-visualization-and-summary-statistics)
  - [ 9. Compare the performance of the algorithms (basic VS boosting VS bagging VS transfer) with respect to your machine learning problem and explain the results](#-9-compare-the-performance-of-the-algorithms-basic-vs-boosting-vs-bagging-vs-transfer-with-respect-to-your-machine-learning-problem-and-explain-the-results)


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

### <a id="implement-algorithms-section-2"></a> b. Decision trees

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

Support vector machines performs well in the high-dimensional feature spaces created by one-hot encoding and often achieves strong accuracy when appropriately regularized. At prediction time it is efficient and reasonably memory-friendly, since the decision function only depends on a subset of the training points (the support vectors). Training can be slow on very large datasets, but for our dataset size this trade-off is acceptable.

<div style="page-break-after: always;"></div>

### <a id="implement-algorithms-section-5"></a> e. Neural networks

**How it works**

A feed forward neural network stacks linear layers with nonlinear activations and learns parameters by backpropagation. For multi class outputs it ends with a softmax layer to produce probabilities.

**Why we chose it**

Our probmlem is a multi class classification problem with three categories to categorize. As a result we chose to implement our neural network with a MLPClassifier from sklearn as they are designed for mulit class problems. Furthermore, neural nets can also learn complex interactions amone demographic, financial, and academic features that simpler linear models might miss. As a result we chose neural network to be among the machine learning models to implement.


<div style="page-break-after: always;"></div>

## <a id="6-compare-performance"></a> 6. Compare the performance of the five algorithms with respect to your problem, explain the results

<p align="center">
<img src="img/comparison_cm_logreg.png" width="400"/><br>
<em>Figure 13: Confusion matrix for logistic regression</em>
</p>

<p align="center">
<img src="img/comparison_fi_logreg.png" width="800"/><br>
<em>Figure 14: Feature importances for logistic regression</em>
</p>

In figure 13, we can se the confusion matric of our logistic regression model. It achieved a accuracy of 75.7% and it correctly predicted a majority of the students that Graduate. However, it performed worse in regards to predicting dropout, but was still accurate in most cases. When predicting students that would still be enrolled beyond normal completion time it misses 50% of the time. One important note for this model is that it is the most accurate of the basic models when predicting the enrolled category.

As seen in figure 14 we can see that the most important feature for logistic regression in determining classifications were the different Cerricular features. How many cerriculars units a student has gotten approved, enrolled in and what grade was achieved, were more importnate for making predictions.

Two course classes are also in the top ten feature importances, which is expected because different cources will have different dropout/enrolled/graduate ratios.



<p align="center">
<img src="img/comparison_cm_dt.png" width="400"/><br>
<em>Figure 15: Confusion matrix for decision tree</em>
</p>

<p align="center">
<img src="img/comparison_fi_dt.png" width="800"/><br>
<em>Figure 16: Feature importances for decision tree</em>
</p>

Our decision tree model achieved a 67.9% accuracy and in figure 15 is the confusion matrix for the model. When predicting dropout it performed about the same as the logistic regression model, but performed slightly worse when predicting graduate. Lastly it also performed worse when predicting enrolled. 

In figure 16 we can see that like logistic regression, the decision tree model relied mostly on the different cerricular features when making its predictions. It had the tuition fees up to date feature, and a course code feature in it's top ten features. Intrestingly it had age at enrollment in it's top ten features importanses.

<p align="center">
<img src="img/comparison_cm_rf.png" width="400"/><br>
<em>Figure 17: Confusion matrix for random forest</em>
</p>

<p align="center">
<img src="img/comparison_fi_rf.png" width="800"/><br>
<em>Figure 18: Feature importances for random forest</em>
</p>

Our random forest model implementation achieved an accuracy of 78.8%, our best performing model. As seen in figure 17 it is very accurate in predicting students that graduate. However it performes fairly similarly to the previous models, albeit a bit better, when predicting dropouts. When predicting enrolled it performs similarly to the decision tree model, but worse than logistic regression.

Additionally, when we analyze the the top then feature importances for our random forrest model we see similarities to the previous models. Cerricular and tuiton features also appear here, whit the interesting addition of GDP and unemployment rate.

<p align="center">
<img src="img/comparison_cm_svm.png" width="400"/><br>
<em>Figure 19: Confusion matrix for SVM with  RBF kernel</em>
</p>

<p align="center">
<img src="img/comparison_fi_svm.png" width="800"/><br>
<em>Figure 20: Feature importances for SVM with RBF kernel</em>
</p>

In figure 19 we can see the SVM model performes alsoms as well as random forest. It achieved an accuracy of 76.6% and predicts graduate correctly a majority of the time. Furthermore, it also performes well with regard to predicting drop out, but struggles to predict enrolled like most of the previous models.

When we estimated feature importance it revealed that also SVM relies on mostly the same features to make it's predictions. The most important being the carricualar features. In figure 20 we also see, with comaprison to the previous models, the addition of debtor, gender, displaced, scholarchip holder, and a fathers qualification feature as a model's top feature importances.

<p align="center">
<img src="img/comparison_cm_mlp.png" width="400"/><br>
<em>Figure 21: Confusion matrix for MLP</em>
</p>

<p align="center">
<img src="img/comparison_fi_mlp.png" width="800"/><br>
<em>Figure 22: Feature importances for MLP</em>
</p>

Lastly, we have our MLP neural network model. It achieved a 75.9% accuracy and performed well when predicting graduate and dropout. Silarly to previous models it also struggeled with predicting enrolled, as it missed about 50% of the time on this classification.

In our estimated feature importances for our MLP model, we see in figure 22, that tuition fees up to date is the most important feature. Like the other models it also utilized the cerricular features to make predictions. Here we also noticed that it had two previously unseen features in it's top ten feature importances, these were a previous qualifications feature and an application mode feature.
<div style="page-break-after: always;"></div>

## <a id="7-boosting-bagging"></a> 7. Implement boosting and bagging with your choice of base models and explain all the steps

<div style="page-break-after: always;"></div>

## <a id="8-transfer-learning"></a> 8. Implement one instance of transfer learning (find a related bigger dataset online) and explain all the steps

First we looked online to find a dataset related to our student-graduation dataset. We were only able to find a single larger dataset related to student graduation/performance. The dataset we found was Student Performance & Behavior Dataset found on Kaggle at https://www.kaggle.com/datasets/mahmoudelhemaly/students-grading-dataset. 

Unfortunatly this dataset shares very few features with our dataset, but as previously stated it was the only somewhat related and larger dataset we could access. This meant we had to map seemingly correlated features between the dataset based on our own intuiton on what makes sense. The features where we could not make any sensible mappings had to be dropped so only the set of overlapping features between the dataset were used for this transfer learning. 

Since we had to drop the non overlapping features from the new dataset, this meant the pretrained model was trained on a small fraction of the original data. As a result, not even this pretrained model reached a desirable accuraccy. When transfering this pretrained model over to our original dataset (with only overlapping features) the accuracy got even worse. This was probably because what we considered to be the most logical mappings between the dataset features were not very accurate. 

Due to these limitations we consider tranfer learning to be an unsuitable training approach for a model predicting our dataset (given the other dataset we were able to find). 

### <a id="transfer-learning-section-1"></a> a. Explain the bigger dataset with visualization and summary statistics.

Age and gender were features in both datasets so they were kept as they were.

Our main dataset had two features named "Curricular units 1st sem (grade)" & "Curricular units 2nd sem (grade)". We chose to use the "Quizzes_Avg" feature in our found dataset to map to these features based on the intuition that higher average score on quizzes would correspond to higher grades in the student's semesters.  We transformed the value range of this feature from 0-100 to the range 0-20. 

We also chose to map the feature "Family_Income_Level" in our found dataset over to the "Debtor" feature in our main dataset. Choosing families with low income as debtors.

Furthermore, we also mapped the feature "Family_Income_Level" == 'High' in the found dataset over to "Tuition fees up to date" in the original based on the assumption that if a student comes from a high erarning family they would have payed all tuition fees on time.

Finally, the "Grade" feature in the found dataset was chosen to become the "Target" column where A & B grades where mapped to graduate, C to enrolled if the students stress level was above 5 and graduate if it was below, D & F to dropout. This was based on the assumption that the poorer(D & F) grades would be an accurate mapping of dropout students, and the best grades(A & B) would be an accurate mapping to graduate. We also made an assumption that a C grade would be graduate if they were adequatly calm as measured by the self reported lower stress levels. Cosequently, we also thought they would be enrolled if they had higher stress levels indecating they where struggeling to maintain a C grade. We do concede that these assumptions are flawed, but we thougth this to be the most sensible way split the data to maintain a reasonable distribution of the target classes. 

<p align="center">
  <img src="new_dataset_vis/grade_dist.png" width="600"/><br>
  <em>Figure 1: Distribution of grades</em>
</p>
<p align="center">
  <img src="new_dataset_vis/without_splitting.png" width="600"/><br>
  <em>Figure 1: Distribution of target without splitting C grade</em>
</p>

<p align="center">
  <img src="new_dataset_vis/target_dist.png" width="600"/><br>
  <em>Figure 1: Distribution of target with split of C grade</em>
</p>
    

<div style="page-break-after: always;"></div>

## <a id="7-compare-performance"></a> 9. Compare the performance of the algorithms (basic VS boosting VS bagging VS transfer) with respect to your machine learning problem and explain the results

<div style="page-break-after: always;"></div>
