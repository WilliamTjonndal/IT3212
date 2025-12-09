# IT3212 Assignment 1: Deep learning and unsupervised learning

## Table of Contents

- [Task 1](#task-1)
  - [Pick any image based dataset from the list, implement the preprocessing and justify the preprocessing steps, extract features and justify the methods used, select features and justify the methods used. Some of this is done already in one of the previous assignments. You can reuse things](#task-1-a)
  - [Implement (using the selected features) one basic machine learning algorithm for classification and justify your choice 20 (without justification 10).](#task-1-b)
  - [Implement (using the selected features) one advanced machine learning algorithm for classification and justify your choice 20 (without justification 10).](#task-1-c)
  - [Implement a CNN with hyperparameter tuning (for this you can directly use the data after the preprocessing) (30)](#task-1-d)
  - [Compare and Explain the results in terms of both the computation time and the performance of the classification algorithms. (30)](#task-1-e)
- [Task 2](#task-2)
  - [Pick any dataset from the list, implement the preprocessing and justify the preprocessing steps,extract features and justify the methods used, select features and justify the methods used. Some of this is done already in one of the previous assignments. You can reuse things.](#task-2-a)
  - [Implement three clustering methods out of the following and justify your choices (30)](#task-2-b)
    - [K-means](#k-means)
    - [Hierarchical clustering](#hierarchical-clustering)
    - [Fuzzy C-means](#fuzzy-c-means)
    - [DBSCAN](#dbscan)
    - [Gaussian mixture models](#gaussian-mixture-models)
    - [Self-organizing maps](#self-organizing-maps)
  - [Compare and Explain the results (30)](#task-2-c)

<div style="page-break-after: always;"></div>

## <a id="task-1"></a> Task 1

### <a id="task-1-a"></a> Pick any image based dataset from the list, implement the preprocessing and justify the preprocessing steps, extract features and justify the methods used, select features and justify the methods used. Some of this is done already in one of the previous assignments. You can reuse things

#### Preprocessing

We have chosen the Intel Image classification dataset. It contains natural scene images labeled into six categories: buildings, forest, glacier, mountain, sea, and street. These are the targets our models aim to predict. The preprocessing stage combined algorithms for detecting corrupted or low-quality images with manual inspection to ensure data quality. This approach minimized the risk of incorrectly discarding valid training images.

**Duplicate Images**

We detected exact duplicates by computing an MD5 hash of each image’s raw pixel values and grouped images with identical hashes. The resulting pairs are shown in figure 1 and 2.

On closer inspection, some duplicate images occur in different categories (e.g., mountain/glacier and building/street). We treat these as intentional overlaps since mountains naturally contain glaciers, and buildings often appear in street scenes. Therefore, these duplicates were kept. However, duplicates within the same category folder (e.g., forest, sea, street) are obviously redundant and was removed.

<p align="center">
  <img src="/assignment4/task1/results/exact_duplicates_same_label/exact_duplicates_same_label_pairs.png" width="500"/><br>
  <em>Figure 1: Exact duplicates with same category detected in the training set</em>
</p>

<p align="center">
  <img src="/assignment4/task1/results/exact_duplicates_diff_label/exact_duplicates_diff_label_pairs.png" width="500"/><br>
  <em>Figure 2: Exact duplicates with different categories detected in the training set</em>
</p>

In addition, we identified perceptual duplicates by computing a perceptual hash for each image and grouping images with identical hashes as visually redundant. Unlike exact duplicates, which rely on MD5 and only catch bit-for-bit identical files, perceptual hashing groups images that look the same, even if they differ slightly in encoding or minor edits. The results are shown in figure 3 and 4, where many image pairs are nearly indistinguishable to the human eye, though some differ slightly in lighting or saturation.

This method identified some of the same images as the exact-duplicate search, but also uncovered images that had been slightly modified. The perceptual-duplicate images which occurs within the same category folder, were removed to eliminate redundant information.

<p align="center">
  <img src="/assignment4/task1/results/perceptual_duplicates_same_label/perceptual_duplicates_same_label_pairs.png" width="500"/><br>
  <em>Figure 3: Perceptual duplicates with same category detected in the training set</em>
</p>
<p align="center">
  <img src="/assignment4/task1/results/perceptual_duplicates_diff_label/" width="500"/><br>
  <em>Figure 4: Perceptual duplicates with different categories detected in the training set</em>
</p>

**Not recognizable images**

We also examined whether the dataset contained blurry, empty/low-edge, almost constant, or overly noisy images. Empty/low-edge images were detected using edge detectors to find cases with very few visible structures, almost constant images were identified by measuring how little the pixel intensities vary, blurry images were found by checking for a lack of fine detail using a Laplacian-based sharpness measure, and overly noisy images were characterized by excessively strong high-frequency responses.

We only identified a small number of blurry and empty/low-edge images (Figures 3 and 4), but visual inspection showed that they are still sufficiently clear and structured to represent their categories, so we kept them in the training set.

<p align="center">
  <img src="task1/results/blurry_examples/blurry_examples.png" width="500"/><br>
  <em>Figure 5: Blurry images detected in the training set</em>
</p>

<p align="center">
  <img src="task1/results/empty_low_edge_examples/empty_low_edge_examples.png" width="500"/><br>
  <em>Figure 6: Empty/Low edge images detected in the training set</em>
</p>

**Misplaced Images**

We also used a modified k-nearest-neighbors algorithm on HSV color-histogram features to identify potentially misplaced images, flagging those whose nearest neighbors mostly shared a different, but mutually consistent, class label. We used the following parameters:

- k: Number of nearest neighbors to examine for each image (excluding the image itself)
- Neigboor difference threshold: Minimum fraction of neighbors that must have a different label than the image to flag it as suspicious
- Minimum alternative fraction: Among the disagreeing neighbors, the minimum fraction that must agree on one specific alternative class. This prevents flagging images where neighbors are split between multiple different classes, ensuring the algorithm only flags images where there's strong consensus on what the correct label should be.

Figure 7 shows the number of misplaced image detected by the algorithm for all categories. Figure 8 to 13 shows some 40 images for each category that the algorithm classified as misplaced, which reveal that there very many potensially misplaced images in the glacier category. We inspected this folder manually.

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
  <em>Figure 9: Potentially misplaced images in forest category/em>
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

### <a id="task-1-b"></a> Implement (using the selected features) one basic machine learning algorithm for classification and justify your choice 20 (without justification 10).

### <a id="task-1-c"></a> Implement (using the selected features) one advanced machine learning algorithm for classification and justify your choice 20 (without justification 10).

### <a id="task-1-d"></a> Implement a CNN with hyperparameter tuning (for this you can directly use the data after the preprocessing) (30)

### <a id="task-1-e"></a> Compare and Explain the results in terms of both the computation time and the performance of the classification algorithms. (30)

<div style="page-break-after: always;"></div>

## <a id="task-2"></a> Task 2

### <a id="task-2-a"></a> Pick any dataset from the list, implement the preprocessing and justify the preprocessing steps,extract features and justify the methods used, select features and justify the methods used. Some of this is done already in one of the previous assignments. You can reuse things.

### <a id="task-2-b"></a> Implement three clustering methods out of the following and justify your choices (30)

#### <a id="hierarchical-clustering"></a> Hierarchical clustering

#### <a id="fuzzy-c-means"></a> Fuzzy C-means

#### <a id="dbscan"></a> DBSCAN

#### <a id="gaussian-mixture-models"></a> Gaussian mixture models

#### <a id="self-organizing-maps"></a> Self-organizing maps

### <a id="task-2-c"></a> Compare and Explain the results (30)
