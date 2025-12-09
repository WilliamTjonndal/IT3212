# IT3212 Assignment 4: Deep learning and unsupervised learning

## Table of Contents

- [IT3212 Assignment 4: Deep learning and unsupervised learning](#it3212-assignment-4-deep-learning-and-unsupervised-learning)
  - [Table of Contents](#table-of-contents)
  - [ Task 1](#-task-1)
    - [ Pick any image based dataset from the list, implement the preprocessing and justify the preprocessing steps, extract features and justify the methods used, select features and justify the methods used. Some of this is done already in one of the previous assignments. You can reuse things](#-pick-any-image-based-dataset-from-the-list-implement-the-preprocessing-and-justify-the-preprocessing-steps-extract-features-and-justify-the-methods-used-select-features-and-justify-the-methods-used-some-of-this-is-done-already-in-one-of-the-previous-assignments-you-can-reuse-things)
    - [ Implement (using the selected features) one basic machine learning algorithm for classification and justify your choice.](#-implement-using-the-selected-features-one-basic-machine-learning-algorithm-for-classification-and-justify-your-choice)
    - [Random forest](#randomforest)
    - [ Implement (using the selected features) one advanced machine learning algorithm for classification and justify your choice.](#-implement-using-the-selected-features-one-advanced-machine-learning-algorithm-for-classification-and-justify-your-choice)
    - [XGBoost](#xgboost)
    - [ Implement a CNN with hyperparameter tuning (for this you can directly use the data after the preprocessing)](#-implement-a-cnn-with-hyperparameter-tuning-for-this-you-can-directly-use-the-data-after-the-preprocessing)
    - [ Compare and Explain the results in terms of both the computation time and the performance of the classification algorithms.](#-compare-and-explain-the-results-in-terms-of-both-the-computation-time-and-the-performance-of-the-classification-algorithms)
  - [ Task 2](#task-2)
    - [ Pick any dataset from the list, implement the preprocessing and justify the preprocessing steps,extract features and justify the methods used, select features and justify the methods used.](#-pick-any-dataset-from-the-list-implement-the-preprocessing-and-justify-the-preprocessing-stepsextract-features-and-justify-the-methods-used-select-features-and-justify-the-methods-used)
    - [ Implement three clustering methods out of the following and justify your choices](#-implement-three-clustering-methods-out-of-the-following-and-justify-your-choices)
      - [ K-means](#k-means)
      - [ Fuzzy C-means](#fuzzy-c-means)
      - [ Gaussian mixture models](#gaussian-mixture-models)
    - [ Compare and Explain the results](#-compare-and-explain-the-results)
      - [ K-means](#compare-k-means)
      - [ Fuzzy C-means](#compare-fuzzy-c-means)
      - [ Gaussian mixture models](#compare-gaussian-mixture-models)


<div style="page-break-after: always;"></div>

## <a id="task-1"></a> Task 1

### <a id="task-1-a"></a> Pick any image based dataset from the list, implement the preprocessing and justify the preprocessing steps, extract features and justify the methods used, select features and justify the methods used. Some of this is done already in one of the previous assignments. You can reuse things

#### Preprocessing

We have chosen the Intel Image classification dataset. It contains natural scene images labeled into six categories: buildings, forest, glacier, mountain, sea, and street. These are the targets our models aim to predict. The preprocessing stage combines algorithms for detecting corrupted or low-quality images with manual inspection to ensure data quality. This approach minimized the risk of incorrectly discarding valid training images.

**Duplicate Images**

We detected exact duplicates by computing an MD5 hash of each image’s raw pixel values and grouped images with identical hashes. The resulting pairs are shown in figure 1 and 2.

On closer inspection, some duplicate images occur in different categories (e.g., mountain/glacier and building/street). We treat these as intentional overlaps since the images of mountains we found also contain glaciers, and the buildings appeared in street scenes. Therefore, these duplicates were kept. However, duplicates within the same category folder (forest, sea, street) are obviously redundant and were removed.

<p align="center">
  <img src="task1/results/exact_duplicates_same_label/exact_duplicates_same_label_pairs.png" width="500"/><br>
  <em>Figure 1: Exact duplicates with same category detected in the training set</em>
</p>

<p align="center">
  <img src="task1/results/exact_duplicates_diff_label/exact_duplicates_diff_label_pairs.png" width="500"/><br>
  <em>Figure 2: Exact duplicates with different categories detected in the training set</em>
</p>

In addition, we identified perceptual duplicates by computing a perceptual hash for each image and grouping images with identical hashes as visually redundant. Unlike exact duplicates, which rely on MD5 and only catch bit-for-bit identical files, perceptual hashing groups images that look the same, even if they differ slightly in encoding or minor edits. The results are shown in figure 3 and 4, where many image pairs are nearly indistinguishable to the human eye, though some differ slightly in lighting or saturation.

This method identified some of the same images as the exact-duplicate search, but also uncovered images that had been slightly modified. The perceptual-duplicate images which occurs within the same category folder, were removed to eliminate redundant information.

<h2 style="color: green;">TODO: Nevne at vi fjerner perceptual duplicates, fordi vi ønsker å kontrollere data augmentation selv
- Refere til bilder og ta med eksakte tall.
</h2>

<p align="center">
  <img src="task1/results/perceptual_duplicates_same_label/perceptual_duplicates_same_label_pairs.png" width="500"/><br>
  <em>Figure 3: Perceptual duplicates with same category detected in the training set</em>
</p>
<p align="center">
  <img src="task1/results/perceptual_duplicates_diff_label/perceptual_duplicates_diff_label_pairs.png" width="500"/><br>
  <em>Figure 4: Perceptual duplicates with different categories detected in the training set</em>
</p>

**Not recognizable images**

We also examined whether the dataset contained blurry, empty/low-edge, almost constant, or overly noisy images. Empty/low-edge images were detected using edge detectors to find cases with very few visible structures, almost constant images were identified by measuring how little the pixel intensities vary, blurry images were found by checking for a lack of fine detail using a Laplacian-based sharpness measure, and overly noisy images were characterized by excessively strong high-frequency responses.

We only identified a small number of blurry and empty/low-edge images (Figures 5 and 6), but visual inspection showed that they are still sufficiently clear and structured to represent their categories, so we kept them in the training set.

<h2 style="color: green;">TODO: nevne at vi ønsker å se etter bilder som er blurry, har få edges, noisy etc og hvorfor vi gjorde? hva vi hadde ønsket å finne
- bildene er tåkete
</h2>

<p align="center">
  <img src="task1/results/blurry_examples/blurry_examples.png" width="500"/><br>
  <em>Figure 5: Blurry images detected in the training set</em>
</p>

<p align="center">
  <img src="task1/results/empty_low_edge_examples/empty_low_edge_examples.png" width="500"/><br>
  <em>Figure 6: Empty/Low edge images detected in the training set</em>
</p>

**Misplaced Images**

We also used a modified k-nearest-neighbors algorithm on color-histogram features to identify potentially misplaced images, flagging those whose nearest neighbors mostly shared a different, but mutually consistent, class label. We used the following parameters:

- k: Number of nearest neighbors to examine for each image (excluding the image itself)
- Neighbour difference threshold: Minimum fraction of neighbors that must have a different label than the image to flag it as suspicious
- Minimum alternative fraction: Among the disagreeing neighbors, the minimum fraction that must agree on one specific alternative class. This prevents flagging images where neighbors are split between multiple different classes, ensuring the algorithm only flags images where there's a strong consensus on what the correct label should be.

Figure 7 shows the number of misplaced images detected by the algorithm for all categories. Figure 8 to 13 shows 48 images for each category that the algorithm classified as misplaced, which reveal that there are many potentially misplaced images in the glacier category. After looking at the first fews potentially misplaced images in each category, we saw that the glacier class contained the most misplaced errors, while the other categories were mostly correct. Since they were only 48 suspicious images in the glacier category, we manually looked through these wrongly classified images.

<h2 style="color: green;">TODO: 
- kommenterer bildene
- nevn valgte hyperparametere
- forklar hvorfor vi ikke valgte prøvde flere hyperparametere
</h2>

<p align="center">
  <img src="task1/results/knn_table.png" width="500"/><br>
  <em>Figure 7: Potentially misplaced images detected by K-Nearest-Neighboor</em>
</p>

<p align="center">
  <img src="task1/suspicious_plots/buildings_suspicious.png" width="500"/><br>
  <em>Figure 8: Potentially misplaced images in buildings category</em>
</p>

<p align="center">
  <img src="task1/suspicious_plots/forest_suspicious.png" width="500"/><br>
  <em>Figure 9: Potentially misplaced images in forest category</em>
</p>

<p align="center">
  <img src="task1/suspicious_plots/glacier_suspicious.png" width="500"/><br>
  <em>Figure 10: Potentially misplaced images in glacier category</em>
</p>

<p align="center">
  <img src="task1/suspicious_plots/mountain_suspicious.png" width="500"/><br>
  <em>Figure 11: Potentially misplaced images in mountain category</em>
</p>

<p align="center">
  <img src="task1//suspicious_plots/sea_suspicious.png" width="500"/><br>
  <em>Figure 12: Potentially misplaced images in sea category</em>
</p>

<p align="center">
  <img src="task1/suspicious_plots/street_suspicious.png" width="500"/><br>
  <em>Figure 13: Potentially misplaced images in street category</em>
</p>

#### Extract and Select Features

<h2 style="color: green;">TODO: 
- legg til intro
- forklaring på hva, hvorfor, hvordan for alle feature extraction methodene
- only data aug for CNN, not for HOG, LBP
</h2>

### <a id="task-1-b"></a> Implement (using the selected features) one basic machine learning algorithm for classification and justify your choice.

<h2 style="color: green;">TODO: 
- legg til intro
- forklaring på hva, hvorfor, hvordan randomforest, svm
</h2>

### RandomForest

<p align="center">
    <img src="task1/img/output.png" width="700"/>
</p>

| Label example      | Meaning                              |
| ------------------ | ------------------------------------ |
| **HOG-9**      | HOG with 9 orientations   |
| **HOG-16**     | HOG with 16 orientations  |
| **LBP-8**      | LBP with 8 points              |
| **LBP-10**     | LBP with 10 points             |
| **HOG-9 + LBP-8** | HOG with 16 bins + LBP with 8 points |

The parameter sweep and resulting accuracy plot clearly show that **HOG features consistently outperform LBP features** when used independently with a RandomForest classifier. HOG with either 9 or 16 orientation bins produces test accuracies in the range of **0.62–0.63**, which is notably higher than the LBP configurations, which remain around **0.54–0.55** regardless of whether 8 or 10 sampling points are used. This difference reflects the fact that HOG captures richer gradient-based spatial structure, edges, shapes, and contours, while LBP focuses primarily on local texture micro-patterns.

LBP on its own underperforms because RandomForests tend to benefit from moderately high-dimensional, discriminative features that capture variation at different spatial scales, whereas LBP produces relatively coarse binary patterns that emphasize uniform local texture. Even with different P values (8 vs. 10 sampling points), the performance remains tightly clustered around 0.54–0.55, indicating that changing the radius or number of neighbors does not significantly increase discriminative power for this dataset. This suggests that the dataset’s class boundaries are not strongly explained by micro-textures alone, and LBP’s invariance properties may also reduce useful variation that the classifier could exploit.

The combined **HOG+LBP** features perform between the individual methods: better than LBP alone, but not always exceeding HOG alone. Their accuracies cluster around **0.60–0.65**, with the best combination (HOG-9 + LBP-8) reaching the highest overall accuracy of roughly **0.647**. This indicates that LBP contributes some complementary information, but not enough to consistently improve upon HOG alone. RandomForests may also struggle with the increased dimensionality when HOG and LBP are concatenated, especially if some dimensions are redundant or noisy. Overall, the results show that HOG is the most useful individual descriptor, while combining it with LBP can offer moderate improvements but is not uniformly beneficial across parameter settings.

<h2 style="color: green;">TODO: 
- forklar hva p values er
- nevn at vi bruker LBP Histogram, i stedet for LBP
</h2>

### <a id="task-1-c"></a> Implement (using the selected features) one advanced machine learning algorithm for classification and justify your choice.

<h2 style="color: green;">TODO: 
- legg til intro
- forklaring på hva, hvorfor, hvordan for xgboost og stacking
</h2>

# XGBoost

<p align="center">
    <img src="task1/img/output1.png" width="1000"/>
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

The HOG-only feature extraction results show that using 9 orientations outperforms 16 orientations (65.3% vs. 62.0%). This suggests that increasing the number of orientations beyond a certain point might add noise to our features, reducing model generalization. The simpler configuration with fewer orientations is enough to capture key shape and edge features relevant for classifying these image classes. Thus, a moderate HOG parameter setting helps maintain good performance without unnecessary complexity.

Effect of LBP parameters on accuracy

The LBP-only features yield lower accuracy overall compared to HOG, with 8 sampling points performing better than 10 points (52.0% vs. 48.7%). Increasing the number of points in LBP may introduce more local texture detail but can also increase noise or variability, which might reduce classifier accuracy. LBP is known to capture fine texture patterns, but for this dataset and XGBoost model, simpler LBP parameters seem more effective than higher complexity.

Combining HOG and LBP features

Combining HOG and LBP features consistently improves accuracy over either method alone. The highest accuracy (71.3%) is achieved with HOG-9 orientations combined with LBP-10 points, demonstrating complementary benefits of capturing both shape and texture information. Interestingly, increasing HOG orientations to 16 while combining with LBP yields slightly lower accuracy, reflecting the previous trend that higher HOG complexity is less helpful. Overall, feature fusion provides richer information for XGBoost to leverage, significantly boosting classification performance across the diverse image classes.

This analysis highlights how tuning feature extraction parameters impacts model accuracy, balancing complexity and representational richness for optimal image classification.

<h2 style="color: green;">TODO: 
- nevn at vi bruker mye enklere versjoner av modelene på 10% av data slik at feature parameter sweep ikke tar uendelig mye tid.
- nevn at vi har brukt hyperparameter tuning på randomforest
</h2>

### <a id="task-1-d"></a> Implement a CNN with hyperparameter tuning (for this you can directly use the data after the preprocessing)

Convolutional Neural Networks (CNNs) are a class of deep learning models specifically designed to exploit the spatial structure in image data. Instead of treating each pixel as an independent feature (as in traditional machine learning models), CNNs use convolutional filters and pooling operations to learn hierarchical feature representations directly from the raw image. This makes them particularly well suited for image classification, compared to models such as Random Forests, XGBoost, or stacking ensembles which typically rely on hand-crafted and/or pre-computed features.

In our experiments, we implemented a CNN in TensorFlow/Keras and trained it on the preprocessed image data before feature extraction. The model consisted of several convolutional and max-pooling layers followed by fully connected layers and a final softmax output over the six classes. To increase robustness and enlarge the effective training set, we applied data augmentation to the training images (horizontal flipping, affine “skewing”, and central cropping followed by resizing). Importantly, this augmentation step was the only additional preprocessing performed for the CNN; we did not perform separate feature extraction as we did for the basic and advance models.

The baseline CNN (with a fixed architecture and reasonable default hyperparameters) trained on the full augmented dataset in approximately 15 minutes on CPU. To investigate the effect of hyperparameters, we then performed a grid search over multiple CNN configurations. This hyperparameter tuning was substantially more expensive: even when using only 10% of the training data, the grid search took more than two hours to complete. The tuned model also showed clear signs of overfitting: it achieved a validation accuracy of 67.8%, while the final test accuracy dropped to 61%. This suggests that the hyperparameter search found a configuration that fit the validation split too closely, without improving generalization to unseen data.

By contrast, when we trained our “standard” CNN model (without the heavy grid search) on 100% of the available data, we obtained a substantially higher test accuracy of 83%. This underlines two important points: (i) CNNs can leverage larger amounts of raw image data effectively thanks to their ability to learn features end-to-end, and (ii) hyperparameter tuning must be done carefully to avoid overfitting to a particular validation set, especially when the tuning budget is large compared to the size of the dataset.

<h2 style="color: green;">TODO: 
- 3. avsnitt: skriv om det vi først satt opp en CNN med default hyperparametere, og fikk 83% accuracy som tok ca. 23 min
- så gjorde vi en GridSearch og tok beste resultat
- nevn hvilke parametere de blir gjort gridsearch på
- forklar hvorfor vi valgte hver metode i data augmentation
</h2>

### <a id="task-1-e"></a> Compare and Explain the results in terms of both the computation time and the performance of the classification algorithms.

When comparing the different classification algorithms, both computation time and predictive performance showed clear differences between the CNN and the traditional machine learning models (Random Forest, XGBoost, and the stacking ensemble).

For the basic and advanced models, we did not feed raw images directly. Instead, we first computed feature representations for each image. This feature extraction pipeline took roughly 30 minutes to run. On top of this, training the basic and advanced models on just 20% of the data exceeded the total training time of the CNN on the full dataset, even with data augmentation.

There are several plausible reasons why Random Forest, XGBoost, and the stacking ensemble required more computation time than the CNN:

Extra preprocessing cost:
The feature extraction step for RF and XGBoost is a separate stage that must be applied to every image before training. In contrast, the CNN learns features directly from the raw pixels, and the only additional preprocessing, data augmentation, is done before its given to the model.

Algorithmic differences:
Tree-based methods build many decision trees. Each tree involves repeated splitting of the data based on feature values, which is relatively expensive on CPUs. A CNN, on the other hand, mainly convolutions and matrix multiplications, which libraries like TensorFlow optimize heavily, using parallelization.

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

Overall, the results show that despite the common perception that deep learning models are always slower and more resource-intensive, a reasonably sized CNN can be competitive or even faster than traditional methods when using GPU. This is especially the case when traditional methods depend on expensive feature extraction pipelines. At the same time, the CNN achieved clearly superior classification performance on this image dataset, which is consistent with its architectural advantages for image-based tasks.

<h2 style="color: green;">TODO: 
- legg til SVM in sammenligningen
</h2>

<div style="page-break-after: always;"></div>

## <a id="task-2"></a> Task 2

### <a id="task-2-a"></a> Pick any dataset from the list, implement the preprocessing and justify the preprocessing steps, extract features and justify the methods used, select features and justify the methods used.

We picked the social media dataset for clustering. The dataset contains data about online news, such as categories they fit into, sentiment analysis, and their popularity. 

An actual use case for clustering on this dataset is to group online news together with others that are similar, preferring to show users that engage more with news from a particular cluster other news within the same cluster. By showing users news similar to those they usually engage with, it's likely total engagement and user retention would increase.

To start off with preprocessing the dataset, we looked at all the columns to get an understanding of what the data represents. The dataset contains several columns that are not useful for our selected use case. The first we removed was `url`, as it's useless for clustering. This is because it's unique and categorical, making it impossible to create clusters from.

We also decided to remove all time-related columns, being `weekday_is_monday`, ..., `weekday_is_sunday`, `is_weekend`, and `timedelta`. The reason for deleting these columns even though they might be useful for clustering is that using clustering with them don't make much sense given our theoretical use case. When recommending similar online news to what you engage with, it wouldn't make sense to take into account which day the news were posted, as recommendations on online platforms should almost always prefer recent news. It doesn't matter what day something was posted if you're recommending something posted within ~2 days anyway.

Using this logic, it might make sense to include `timedelta`, which refers to the days between the article publication and dataset aquisition. The reason we didn't include this column is because this value has a large range, up to several years. This doesn't fit well with our goal, as a recommendation algorithm should almost never recommend articles as old as that. If we were to create a recommendation algorithm based on our clustering results, we would put an external limitation that would heavily prefer recent news instead of including age in our clustering. 

`timedelta` does have an effect on `shares`, as older articles have more time to accumulate shares, but we still chose to includes `shares` when clustering. This is because all data points have a `timedelta` of at least 8 days, so we think all articles have had some time to get a number of shares that would be highly correlated with the number of shares they would have after a few days (This assumes articles gain the most traction/shares when they are recently released, meaning even if older articles have much more time to gain shares, most shares are gained within the first few days of release).

<h2 style="color: green;">TODO: 
- forklar litt bedre timedelta
</h2>

#### Scaling
After removing the columns we don't want to include in clustering, it's time to scale the data. It's important to scale our data so the features with a larger range of values won't be preferred over those with smaller ranges based only on their larger range. Looking at the distribution of the data in figure #, we can see columns referring to shares, like `shares` and `kw_avg_max` (the average of the max amount of shares for articles assigned each keyword) have a much larger range than the rest, meaning they would be likely to overpower the other features. Scaling the data will make all our features have the same scale so each feature's importance will be decided fairly.

<p align="center">
<img src="task2/img/scaling_dist_before.png" width="600"/><br>
<em>Figure #: Distribution of data before scaling</em>
</p>

We chose to use min-max scaling, mostly because it's results are easier to understand for columns like `shares` and `num_imgs`, and it preserves the distribution of our data. The results being easier to understand is not really the case for columns representing sentiment analysis, like `global_sentiment_polarity` and `title_subjectivity`, as we don't have an intuitive understanding of what a specific value means, other than in relation to other values. We still chose to use min-max scaling here to keep the same scaling method for all our features, and again to preserve the distribution of all our features. Looking at the distribution of the scaled data in figure #, we can see the distributions now look much more even which should give better results for the clustering. 

<p align="center">
<img src="task2/img/scaling_dist_after.png" width="600"/><br>
<em>Figure #: Distribution of data after scaling</em>
</p>

#### Outlier detection
When removing outliers from our data, we chose between z-score and IQR. We decided to use IQR as we can see in figure # that the distribution of almost every column is not normally distributed, but skewed. After some testing, we noticed removing outliers using IQR on all non-categorical columns would remove way more rows than expected. To keep a larger portion of the dataset, we had to select which features to use for outlier detection and removal.

We decided not to use outlier detection on columns referencing shares or sentiment analysis, as shares are more extremely skewed than other features in the dataset. Features referencing shares include `shares` and `kw_avg_max`. We also decided not to remove outliers using features based on sentiment analysis, as it would be very hard for us to tell if a very high or low value is actually outside the range of what's likely a real data point. Detecting outliers based on a column we don't know a real range of would not be a good idea, as the goal of handling outliers is removing or changing values not generated by the same method as the others.

Performing outlier detection on the remaining columns (not categorical, referencing shares, or based on sentiment analysis), such as `n_tokens_title` and `num_videos`, we were able to find `2360` outliers. Looking at some of these outliers, they contain things like there being `91` videos or `116` images in an article. This is about 6% of our dataset, which has a total of `39644` rows. We decided to remove the data points containing the outliers, as it's not a very large portion of our dataset.

In figure #, you can see the distribution of our features after removing outliers, while figure # shows the distribution after re-scaling our dataset between 0 and 1. We did outlier detection after scaling to better visualize our results, but it has the same effect as detecting outliers before scaling when using min-max scaling, as the distribution within each feature stays the same.

<h2 style="color: green;">TODO: 
- hvis noen numeriske distribusjon før og etter outlier detection
- en setning om det ikke er noe manglende data
- screenshots fra VScode
- forklaring alle kolonne, kan ha en paragraf for kolonnene som er lignende
</h2>

<p align="center">
<img src="task2/img/iqr_dist.png" width="600"/><br>
<em>Figure #: Distribution of data after removing outliers</em>
</p>

<p align="center">
<img src="task2/img/iqr_dist_rescaled.png" width="600"/><br>
<em>Figure #: Distribution of data after removing outliers and re-scaling</em>
</p>

<h2 style="color: green;">TODO: 
- endre figurene til å vises i høyden, en til venstre og en til høyre.
- gjør det samme med alle andre boxplots
</h2>

#### Dimensionality reduction
We decided to reduce the dimensions of our dataset, as we think it will improve the performance of our clustering methods, as well as giving us a better visualization of our data. Our biggest reason for thinking reducing dimensions will give better performance when clustering is that distance between points becomes less useful the more dimensions are used. Due to how euclidian distance is calculated, the distance between every point converges as dimensions increase, meaning the more dimensions there are in the dataset, the less variation there is in the distance between each point. If every point is almost the same distance from eachother, it becomes very hard to seperate them into meaningful clusters. This problem is a bit exaggerated as it has a much more noticable impact with 100+ dimensions, but it's still better to reduce the dimensions to minimize this effect.

For dimensionality reduction, we thought about using PCA and t-SNE. PCA focuses on keeping as much of the variance in the data as possible, while t-SNE tries to keep higher-dimensional neighbors close even in lower dimensions. While t-SNE sound like a good fit for our dataset, it has a problem which makes it unsuitable for our selected use case. t-SNE finds similarities between all points in the dataset, which works well for preserving neighborhoods, but makes no mapping function that can be used for future data points. This means that if we want to add a new data point (such as a new article being created), we would have to redo our dimensionality reduction on all our data. Due to this, we chose to use PCA for dimensionality reduction, keeping enough principal components to preserve 95% of the variance of the data. This leaves us with 22 out of the 51 original features remaining, as seen in figure #.

<p align="center">
<img src="task2/img/pca_95.png" width="400"/><br>
<em>Figure #: Explained variance by principal components</em>
</p>

The other advantage of PCA is being useful for visualizing data. We can see the visualization of the dataset using the first 3 principal components in both 2d, in figure #, and in 3d, in figure #.

<p align="center">
<img src="task2/img/pca_pairplot.png" width="600"/><br>
<em>Figure #: Pairplot of first 3 principal components</em>
</p>

<p align="center">
<img src="task2/img/pca_3d.png" width="600"/><br>
<em>Figure #: 3D plot of first 3 principal components</em>
</p>

### <a id="task-2-b"></a> Implement three clustering methods out of the following and justify your choices

There are several different types of clustering algorithms to choose from, each with different properties. There are several types of clustering algorithms to choose from, but we mainly looked at centroid-based, density-based and distribution-based clustering algorithms.\
The simplest are centroid-based algorithms that create cluster centers and assign data points to clusters based on their distance from the cluster centers. Examples of this are K-means and fuzzy C-means.\
Density-based algorithms define clusters based on the density of data points. This fits well for finding clusters of different shapes, but won't work well for clusters with varying densities. DBSCAN is an example of a density-based clustering algorithm.\
Distribution-based algorithms assume clusters are generated by probability distributions, which work well for clusters of different (but not too complicated) shapes and densities, but it does assume data is generated in a specific distribution. An example of distribution-based clustering algorithms is Gaussian Mixture Models.

#### <a id="k-means"></a> K-means

K-means was chosen mainly because it's the easiest clustering algorithm to understand. A problem with K-means and other centroid based algorithms is that it works best for clusters that are approximately spherical and similar in size, which is not the case for this dataset. This could have a negative impact on the performance of the clustering if the data points don't seperate well with centroid based clustering algorithms.

#### <a id="fuzzy-c-means"></a> Fuzzy C-means

Fuzzy C-means works almost like an improved version of K-means for this dataset. The biggest difference between them is that fuzzy C-means assigns each data point a membership value for every cluster, meaning it would be better for our use case. Even though articles are seperated into clusters, we don't want to make it impossible for articles that fit better in another cluster to be recommended to users, just less likely. Using fuzzy C-means, we can assign how likely an article is to be recommended to users based on its membership value to each cluster. Aside from that, it works very similarly to K-means as both are centroid-based clustering algorithms.

#### <a id="gaussian-mixture-models"></a> Gaussian mixture models

Gaussian Mixture Models were chosen because they offer a more flexible way to cluster the data. As opposed to centroid-based clustering algorithms which assume roughly spherical clusters of similar size, GMM is a distribution-based clustering algorithm that assumes each cluster has a shape and spread in the data, which means clusters don’t have to be perfectly round or all the same size. This fits better for our dataset, as we think it better matches the distribution of the clusters we can visualize from figure #. The reason we chose GMM over other clustering algorithms is that we thought a distribution-based clustering method would be best for our dataset, as clusters can have different shapes and densities, making it more suited to guess .

Like fuzzy C-means, GMM assigns a probability to each article for belonging to every cluster. This way, this clusterign algorithm doens't force articles that could belong to multiple clusters into just one.

A problem shared between all our selected clustering algorithms is that you have to specify the amount of clusters created for each of them. While it would be nice to use a clustering method that doesn't depend on the amount of clusters, we think we were able to find a good value for the amount of clusters to create by seeing how different values look in our 3D visualization. By testing with different values for our amount of clusters using K-means as our baseline model, we decided to go with 4 clusters. This can be seen in figures #-#, and we decided to go with 4 clusters, figure #, because we thought it had the best compromise between splitting into several clusters while keeping them large and seperated enough to justify them being seperate clusters.

<table align="center">
  <tr>
    <td align="center">
      <img src="task2/img/clusters_3.png" width="300"/><br>
      <em>Figure #: K-means clustering with k=3</em>
    </td>
    <td align="center">
      <img src="task2/img/clusters_4.png" width="300"/><br>
      <em>Figure #: K-means clustering with k=4</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="task2/img/clusters_5.png" width="300"/><br>
      <em>Figure #: K-means clustering with k=5</em>
    </td>
    <td align="center">
      <img src="task2/img/clusters_6.png" width="300"/><br>
      <em>Figure #: K-means clustering with k=6</em>
    </td>
  </tr>
</table>

### <a id="task-2-c"></a> Compare and Explain the results

#### <a id="compare-k-means"></a> K-means

<p align="center">
<img src="task2/img/clusters_kmeans.png" width="500"/><br>
<em>Figure #: Visualization of K-means clustering</em>
</p>

<p align="center">
<img src="task2/img/features_kmeans.png" width="600"/><br>
<em>Figure #: Top features contributing to K-means clustering</em>
</p>

#### <a id="compare-fuzzy-c-means"></a> Fuzzy C-means

<p align="center">
<img src="task2/img/clusters_fcm.png" width="500"/><br>
<em>Figure #: Visualization of fuzzy C-means clustering</em>
</p>

<p align="center">
<img src="task2/img/features_fcm.png" width="600"/><br>
<em>Figure #: Top features contributing to fuzzy C-means clustering</em>
</p>

#### <a id="compare-gaussian-mixture-models"></a> Gaussian mixture models

<p align="center">
<img src="task2/img/clusters_gmm.png" width="500"/><br>
<em>Figure #: Visualization of GMM clustering</em>
</p>

<p align="center">
<img src="task2/img/features_gmm.png" width="600"/><br>
<em>Figure #: Top features contributing to GMM clustering</em>
</p>

All our chosen clustering algorithms had very similar results, resulting in almost the exact same clusters. You can see minor differences between them, like the "top" cluster in GMM including more data points further towards the center of the plot than in K-means and fuzzy C-means. This difference is likely due to these data points being more similar to the "top" cluster's distribution, even though many of them are closer to the centers of the other three clusters.

There are even less differences between K-means and fuzzy C-means, but this is expected as they are both very similar centroid-based algorithms. The main difference between them is the addition of membership values, which are not easily seen on this plot even though more unsure data points are lighter in color. Membership values being a part of both fuzzy C-means and GMM will make both of them better suited to our use case than K-means.

We can also see the top features contributing to the selection of clusters by each algorithm. The results here are also very similar to eachother, mainly focusing on categories like `data_channel_is_world` and `LDA_00` (Closeness to LDA topic 0). The weight for each specific feature is different between the algorithms, but there are no major changes.

A lot of the similarity here likely come from how the original features were translated into princial components when doing PCA, and you might see a more varied set of contributiong features if using the base data. Even if the results are very similar and mostly using the same few features to base a data point's cluster on, this actually fits very well for our use case. Splitting articles mainly by categories and topics makes the most sense to create a useful recommendation algorithm. Had our results been that clusters were mainly decided by `shares` and similar features, this would probably not show users articles they are as interested in as with our current main clustering factors.

<h2 style="color: green;">TODO: 
- sjekk og rapporter om det finnes tome eller manglende verdier
- metrikker for å finne cluster performance
- noen eksempler for hver cluster i final results (GMM)
- legg til grafer for fuzzy c mean, og GMM for alle k-verdier
- drøfte at det er vanskelig å velge antall clusters (i clusterseksjonen)
</h2>