# IT3212 Assignment 4: Deep learning and unsupervised learning

## Table of Contents

- [IT3212 Assignment 4: Deep learning and unsupervised learning](#it3212-assignment-4-deep-learning-and-unsupervised-learning)
  - [Table of Contents](#table-of-contents)
  - [ Task 1](#-task-1)
    - [ Pick any image based dataset from the list, implement the preprocessing and justify the preprocessing steps, extract features and justify the methods used, select features and justify the methods used. Some of this is done already in one of the previous assignments. You can reuse things](#-pick-any-image-based-dataset-from-the-list-implement-the-preprocessing-and-justify-the-preprocessing-steps-extract-features-and-justify-the-methods-used-select-features-and-justify-the-methods-used-some-of-this-is-done-already-in-one-of-the-previous-assignments-you-can-reuse-things)
    - [ Implement (using the selected features) one basic machine learning algorithm for classification and justify your choice.](#-implement-using-the-selected-features-one-basic-machine-learning-algorithm-for-classification-and-justify-your-choice)
    - [RandomForest](#randomforest)
    - [ Implement (using the selected features) one advanced machine learning algorithm for classification and justify your choice.](#-implement-using-the-selected-features-one-advanced-machine-learning-algorithm-for-classification-and-justify-your-choice)
- [XGBoost](#xgboost)
    - [ Implement a CNN with hyperparameter tuning (for this you can directly use the data after the preprocessing)](#-implement-a-cnn-with-hyperparameter-tuning-for-this-you-can-directly-use-the-data-after-the-preprocessing)
    - [ Compare and Explain the results in terms of both the computation time and the performance of the classification algorithms.](#-compare-and-explain-the-results-in-terms-of-both-the-computation-time-and-the-performance-of-the-classification-algorithms)
  - [ Task 2](#-task-2)
    - [ Pick any dataset from the list, implement the preprocessing and justify the preprocessing steps,extract features and justify the methods used, select features and justify the methods used. Some of this is done already in one of the previous assignments. You can reuse things.](#-pick-any-dataset-from-the-list-implement-the-preprocessing-and-justify-the-preprocessing-stepsextract-features-and-justify-the-methods-used-select-features-and-justify-the-methods-used-some-of-this-is-done-already-in-one-of-the-previous-assignments-you-can-reuse-things)
    - [ Implement three clustering methods out of the following and justify your choices](#-implement-three-clustering-methods-out-of-the-following-and-justify-your-choices)
      - [ Hierarchical clustering](#-hierarchical-clustering)
      - [ Fuzzy C-means](#-fuzzy-c-means)
      - [ DBSCAN](#-dbscan)
      - [ Gaussian mixture models](#-gaussian-mixture-models)
      - [ Self-organizing maps](#-self-organizing-maps)
    - [ Compare and Explain the results](#-compare-and-explain-the-results)


<div style="page-break-after: always;"></div>

## <a id="task-1"></a> Task 1

### <a id="task-1-a"></a> Pick any image based dataset from the list, implement the preprocessing and justify the preprocessing steps, extract features and justify the methods used, select features and justify the methods used. Some of this is done already in one of the previous assignments. You can reuse things

### <a id="task-1-b"></a> Implement (using the selected features) one basic machine learning algorithm for classification and justify your choice.

### RandomForest

<p align="center">
    <img src="task1/results/randomforest feature sweep.png" width="700"/>
</p>

| Label example      | Meaning                              |
| ------------------ | ------------------------------------ |
| **HOG-9**      | HOG with 9 orientations   |
| **HOG-16**     | HOG with 16 orientations  |
| **LBP-8**      | LBP with 8 points              |
| **LBP-10**     | LBP with 10 points             |
| **HOG-9 + LBP-8** | HOG with 16 bins + LBP with 8 points |

The parameter sweep and resulting accuracy plot clearly show that **HOG features consistently outperform LBP features** when used independently with a RandomForest classifier. HOG with either 9 or 16 orientation bins produces test accuracies in the range of **0.62–0.63**, which is notably higher than the LBP configurations, which remain around **0.54–0.55** regardless of whether 8 or 10 sampling points are used. This difference reflects the fact that HOG captures richer gradient-based spatial structure—edges, shapes, and contours—while LBP focuses primarily on local texture micro-patterns. For a dataset where global shape information is more important than fine-grained texture, HOG will generally provide a more informative feature space for tree-based models.

LBP on its own underperforms because RandomForests tend to benefit from moderately high-dimensional, discriminative features that capture variation at different spatial scales, whereas LBP produces relatively coarse binary patterns that emphasize uniform local texture. Even with different P values (8 vs. 10 sampling points), the performance remains tightly clustered around 0.54–0.55, indicating that changing the radius or number of neighbors does not significantly increase discriminative power for this dataset. This suggests that the dataset’s class boundaries are not strongly explained by micro-textures alone, and LBP’s invariance properties may also reduce useful variation that the classifier could exploit.

The combined **HOG+LBP** features perform between the individual methods: better than LBP alone, but not always exceeding HOG alone. Their accuracies cluster around **0.60–0.65**, with the best combination (HOG-9 + LBP-8) reaching the highest overall accuracy of roughly **0.647**. This indicates that LBP contributes some complementary information, but not enough to consistently improve upon HOG alone. RandomForests may also struggle with the increased dimensionality when HOG and LBP are concatenated, especially if some dimensions are redundant or noisy. Overall, the results show that HOG is the most useful individual descriptor, while combining it with LBP can offer moderate improvements but is not uniformly beneficial across parameter settings.

### <a id="task-1-c"></a> Implement (using the selected features) one advanced machine learning algorithm for classification and justify your choice.

# XGBoost

<p align="center">
    <img src="task1/results/xgboost feature sweep.png" width="1000"/>
</p>

| Method    | HOG Orientations | LBP Points (P) | Test Accuracy |
| --------- | ---------------- | -------------- | ------------- |
| HOG       | 9                | -              | 0.653         |
| HOG       | 16               | -              | 0.620         |
| LBP       | -                | 8              | 0.520         |
| LBP       | -                | 10             | 0.487         |
| HOG + LBP | 9                | 8              | 0.707         |
| HOG + LBP | 9                | 10             | 0.713         |
| HOG + LBP | 16               | 8              | 0.687         |
| HOG + LBP | 16               | 10             | 0.627         |

Effect of HOG parameters on accuracy

The HOG-only feature extraction results show that using 9 orientations outperforms 16 orientations slightly (65.3% vs. 62.0%). This suggests that increasing the number of orientations beyond a certain point might add redundant or noisy information, slightly reducing model generalization. The simpler configuration with fewer orientations is enough to capture key shape and edge features relevant for classifying these image classes. Thus, a moderate HOG parameter setting helps maintain good performance without unnecessary complexity.

Effect of LBP parameters on accuracy

The LBP-only features yield lower accuracy overall compared to HOG, with 8 sampling points performing better than 10 points (52.0% vs. 48.7%). Increasing the number of points in LBP may introduce more local texture detail but can also increase noise or variability, which might reduce classifier accuracy. LBP is known to capture fine texture patterns, but for this dataset and XGBoost model, simpler LBP parameters seem more effective than higher complexity.

Combining HOG and LBP features

Combining HOG and LBP features consistently improves accuracy over either method alone. The highest accuracy (71.3%) is achieved with HOG-9 orientations combined with LBP-10 points, demonstrating complementary benefits of capturing both shape and texture information. Interestingly, increasing HOG orientations to 16 while combining with LBP yields slightly lower accuracy, reflecting the previous trend that higher HOG complexity is less helpful. Overall, feature fusion provides richer information for XGBoost to leverage, significantly boosting classification performance across the diverse image classes.

This analysis highlights how tuning feature extraction parameters impacts model accuracy, balancing complexity and representational richness for optimal image classification.

### <a id="task-1-d"></a> Implement a CNN with hyperparameter tuning (for this you can directly use the data after the preprocessing)

Convolutional Neural Networks (CNNs) are a class of deep learning models specifically designed to exploit the spatial structure in image data. Instead of treating each pixel as an independent feature (as in traditional machine learning models), CNNs use convolutional filters and pooling operations to learn hierarchical feature representations directly from the raw image. This makes them particularly well suited for image classification, compared to models such as Random Forests, XGBoost, or stacking ensembles which typically rely on hand-crafted and/or pre-computed features.

In our experiments, we implemented a CNN in TensorFlow/Keras and trained it on the preprocessed image data. The model consisted of several convolutional and max-pooling layers followed by fully connected layers and a final softmax output over the six classes. To increase robustness and enlarge the effective training set, we applied data augmentation to the training images (horizontal flipping, affine “skewing”, and central cropping followed by resizing). Importantly, this augmentation step was the only additional preprocessing performed for the CNN; we did not perform separate feature extraction as we did for the basic and advance models.

The baseline CNN (with a fixed architecture and reasonable default hyperparameters) trained on the full augmented dataset in approximately 15 minutes on CPU. To investigate the effect of hyperparameters, we then performed a grid search over multiple CNN configurations. This hyperparameter tuning was substantially more expensive: even when using only 10% of the training data, the grid search took more than two hours to complete. The tuned model also showed clear signs of overfitting: it achieved a validation accuracy of 67.8%, while the final test accuracy dropped to 61%. This suggests that the hyperparameter search found a configuration that fit the validation split too closely, without improving generalization to unseen data.

By contrast, when we trained our “standard” CNN model (without the heavy grid search) on 100% of the available data, we obtained a substantially higher test accuracy of 83%. This underlines two important points: (i) CNNs can leverage larger amounts of raw image data effectively thanks to their ability to learn features end-to-end, and (ii) hyperparameter tuning must be done carefully to avoid overfitting to a particular validation set, especially when the tuning budget is large compared to the size of the dataset.

### <a id="task-1-e"></a> Compare and Explain the results in terms of both the computation time and the performance of the classification algorithms.

When comparing the different classification algorithms, both computation time and predictive performance showed clear differences between the CNN and the traditional machine learning models (Random Forest, XGBoost, and the stacking ensemble).

For the basic and advanced models, we did not feed raw images directly. Instead, we first computed feature representations for each image. This feature extraction pipeline took roughly 30 minutes to run. On top of this, training the basic and advanced models on just 20% od the data exceeded the total training time of the CNN on the full dataset, even with data augmentation.

There are several plausible reasons why Random Forest, XGBoost, and the stacking ensemble required more computation time than the CNN:

Extra preprocessing cost:
The feature extraction step for RF and XGBoost is a separate stage that must be applied to every image before training. In contrast, the CNN learns features directly from the raw pixels, and the only additional preprocessing, data augmentation, is done before its given to the model.

Algorithmic differences:
Tree-based methods build many decision trees. Each tree involves repeated splitting of the data based on feature values, which is relatively expensive on CPUs. A CNN, on the other hand, mainly convolutions and matrix multiplications, which libraries like TensorFlow optimize heavily, using paralellization.

Model complexity over multiple models (stacking):
The stacking model combines predictions from several base learners (e.g., Random Forest and SVM) into a meta-model (logistic regression). This effectively multiplies the training cost: each base model must be trained, predictions must be computed, and then the meta-model must be fitted. This multi-stage procedure is naturally more time-consuming than training a single CNN end-to-end. However it performed better than its base learners. Stacking can have the benefit of better accuracy by learning from the predictions of its base learners, thereby making its predictions more reliable. Furthermore, stacking allows the meta model to make use of its base learners strengths, and recognize their mistakes. Lastly, stacking is also highly flexible to different problems. However, as in our case, it is computationally heavy, even with only two base learners. If a problem requires a quick solution and deployment, this strategy would not be adviced.

In terms of classification performance, the CNN clearly outperformed both Random Forest and XGBoost on this image classification task. While the tuned CNN variant overfitted and achieved only 61% test accuracy, our main CNN model trained on 100% of the data reached 83% accuracy, compared to substantially lower accuracies for the tree-based models.

This performance gap can be explained by how the models use information in the data:

Exploiting spatial structure:
CNNs operate directly on 2D image grids with multiple channels and use convolutional filters to capture local patterns such as edges, textures, and shapes. Deeper layers combine these into higher-level concepts. This hierarchical feature learning is very effective for images. We think that maybe the features that the cnn extracts on its own, might be better features compared to the ones we precomputed for our other models.

Information loss in feature extraction:
Furthermore, for Random Forest, XGBoost ans the stacking ensemble, we reduced each image to a small set of hand-crafted features (three feature extractions per image). While this drastically reduces dimensionality and makes the models easier to train, it also loses spatial and textural information present in the original images. If the chosen features are not expressive enough, the models are simply not given sufficient information to match the CNN’s performance.

Model capacity and flexibility:
The CNN has a high capacity to approximate complex decision boundaries directly in pixel space, while the tree-based models are constrained to operate on a small, fixed feature vector. Even powerful ensemble methods like XGBoost will be limited by the quality and richness of those features. In our experiments, this likely led to a situation where the CNN could capture more nuanced visual patterns and therefore generalize better on the test set.

Overall, the results show that, despite the common perception that deep learning models are always slower and more resource-intensive, a reasonably sized CNN can be competitive or even faster than traditional methods in practice—especially when the latter depend on expensive feature extraction pipelines. At the same time, the CNN achieved clearly superior classification performance on this image dataset, which is consistent with its architectural advantages for image-based tasks.


<div style="page-break-after: always;"></div>

## <a id="task-2"></a> Task 2

### <a id="task-2-a"></a> Pick any dataset from the list, implement the preprocessing and justify the preprocessing steps,extract features and justify the methods used, select features and justify the methods used. Some of this is done already in one of the previous assignments. You can reuse things.

### <a id="task-2-b"></a> Implement three clustering methods out of the following and justify your choices


#### <a id="hierarchical-clustering"></a> Hierarchical clustering

#### <a id="fuzzy-c-means"></a> Fuzzy C-means

#### <a id="dbscan"></a> DBSCAN

#### <a id="gaussian-mixture-models"></a> Gaussian mixture models

#### <a id="self-organizing-maps"></a> Self-organizing maps


### <a id="task-2-c"></a> Compare and Explain the results
