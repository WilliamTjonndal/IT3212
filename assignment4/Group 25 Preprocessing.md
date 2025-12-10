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

We use the Intel Image Classification dataset for this task. It contains natural scene images labeled into six categories: buildings, forest, glacier, mountain, sea, and street. These category labels are the targets our models predict. The table below summarizes the number of images in the provided training and test sets. 

<div align="center">

| Category      | Training images | Test images |
|---------------|-----------------|-------------|
| **Buildings** | 2 191           | 437         |
| **Forest**    | 2 271           | 474         |
| **Glacier**   | 2,404           | 553         |
| **Mountain**  | 2 512           | 525         |
| **Sea**       | 2 274           | 510         |
| **Street**    | 2 382           | 501         |
| **Total**     | 14 034          | 3 000       |

</div>

<p align="center"><em>Table 1: Number of training and test images per category.</em></p>

In total, we removed 78 images from the training set, approximately 0.5% of all training images, a negligible proportion that can be discarded without meaningfully reducing the available information. Table 2 summarizes how many images were removed at each preprocessing stage, which will be further described in the following sections.

<div align="center">

| Method                    | Removed images  | Share      |
|---------------------------|-----------------|------------|
| **Exact Duplicates**      | 5               | 0.04%      |
| **Perceptual Duplicates** | 10              | 0.07%      |
| **Miscategorized Images** | 63              | 0.4%       |
| **Total**                 | 78              | 0.6%       |

</div>

<p align="center"><em>Table 2: Number of removed images from training set.</em></p>

Figure 1 and 2 show some sample images from each category in the training and test set.

<p align="center">
  <img src="task1/results/train/train_sample_images_per_class.png" width="500"/><br>
  <em>Figure 1: Sample images for the training set</em>
</p>

<p align="center">
  <img src="task1/results/test/test_sample_images_per_class.png" width="500"/><br>
  <em>Figure 2: Sample images for the test set</em>
</p>

#### Preprocessing

The preprocessing stage combined algorithms for detecting corrupted or low-quality images with manual inspection to ensure data quality. This approach minimized the risk of incorrectly discarding valid training images.

**Duplicate Images**

We detected exact duplicates by computing an MD5 hash (a short, fixed-length “fingerprint” of the image data) of each image’s raw pixel values and grouped images with identical hashes. The resulting duplicates pairs are shown in figure 3 and 4.

On closer inspection, some duplicate images appear in different categories (e.g., mountain/glacier and building/street), as shown in Figure 4. We treat these as intentional overlaps, since mountains can naturally contain glaciers and buildings often appear in street scenes, so these duplicates were kept. However, duplicates within the same category folder (e.g., forest, sea, street), as shown in Figure 3, are clearly redundant and were removed. There were only five such duplicates in the entire training set, which is a negligible fraction of the dataset, so removing them is unlikely to affect the results.

<p align="center">
  <img src="task1/results/exact_duplicates_same_label/exact_duplicates_same_label_pairs.png" width="500"/><br>
  <em>Figure 3: Exact duplicates with same category detected in the training set</em>
</p>

<p align="center">
  <img src="task1/results/exact_duplicates_diff_label/exact_duplicates_diff_label_pairs.png" width="500"/><br>
  <em>Figure 4: Exact duplicates with different categories detected in the training set</em>
</p>

In addition, we identified perceptual duplicates by computing a perceptual hash for each image and grouping images with identical hashes. Unlike exact duplicates, which rely on MD5 and only catch bit-for-bit identical files, perceptual hashing groups images that look the same, even if they differ slightly in encoding or minor edits. The results are shown in figure 5 and 6, where many image pairs are nearly indistinguishable to the human eye, though some differ slightly in lighting or saturation.

This method identified some of the same images as the exact-duplicate search, but also uncovered images that had been slightly modified. Perceptually duplicate images that appeared in multiple categories were retained for the same reasons as the exact duplicates, but those occurring within the same category folder were removed to eliminate redundant information. In total, we removed 10 perceptual duplicates. These duplicates with slight modifications could be seen as intented data augmentation, but we removed them so that we can control the augmentation process ourselves. This will be described further in the next sections.

<p align="center">
  <img src="task1/results/perceptual_duplicates_same_label/perceptual_duplicates_same_label_pairs.png" width="500"/><br>
  <em>Figure 5: Perceptual duplicates with same category detected in the training set</em>
</p>
<p align="center">
  <img src="task1/results/perceptual_duplicates_diff_label/perceptual_duplicates_diff_label_pairs.png" width="500"/><br>
  <em>Figure 6: Perceptual duplicates with different categories detected in the training set</em>
</p>

**Not recognizable images**

We also examined whether the dataset contained blurry, empty/low-edge, almost constant, or overly noisy images. Empty or low-edge images were detected using edge detectors to flag cases with very few visible structures, almost constant images were identified by measuring how little the pixel intensities vary, blurry images were found by checking for a lack of fine detail using a Laplacian-based sharpness measure, and overly noisy images were characterized by excessively strong high-frequency responses.

We applied these checks to identify foggy, grainy, blurry, and otherwise unrepresentative images that would be difficult to classify reliably and could introduce noise into the model. In practice, we found only a small number of blurry and empty/low-edge images shown in figure 7 and 8, and visual inspection showed that they are still sufficiently clear and structured to represent their categories, so we decided to keep them in the training set. 

<p align="center">
  <img src="task1/results/blurry_examples/blurry_examples.png" width="500"/><br>
  <em>Figure 7: Blurry images detected in the training set</em>
</p>

<p align="center">
  <img src="task1/results/empty_low_edge_examples/empty_low_edge_examples.png" width="500"/><br>
  <em>Figure 8: Empty/Low edge images detected in the training set</em>
</p>

**Miscategorized Images**

We also used a modified K-nearest-neighbors algorithm (KNN) with Euclidean distance to identify potentially miscategorized images, flagging those whose nearest neighbors mostly shared a different, but mutually consistent, class label. By miscategorized, we refer to training images that have been assigned the wrong class label. For example a building being labeled as a forest. By suspicious, we mean training images that the algorithm suggests may be incorrectly labeled, but which require manual inspection to confirm.

In detail, each image was represented using an HSV color histogram, where each pixel is defined by its hue (dominant color), saturation (color intensity), and value (brightness), and the histogram captures the number of pixels falling into discrete bins across these three components. This was chosen over RGB because HSV separates color from lighting, making color comparisons more robust to the substantial illumination differences in our data and thus more effective for detecting mislabeled images across the categories. We used the following parameters:

- **K (k)**: Number of nearest neighbors to examine for each image (excluding the image itself). We set this to 20 to obtain a neighborhood that is both local and statistically stable.
- **Neigboor difference threshold (neighbor_diff_threshold)**: Minimum fraction of neighbors that must have a different label than the image to flag it as suspicious. We set this to 80% so that only cases with strong disagreement are flagged.
- **Minimum alternative fraction (min_alt_frac)**: Among the disagreeing neighbors, the minimum fraction that must agree on the same alternative class. We set this to 60% to avoid scattered disagreements and only flag images when there is a clear consensus on a different label.

Figure 9 shows how the KNN-based method flagged a potentially miscategorized image, specifically a flower that had been mislabeled as a glacier (glacier/15039.jpg). In the 3D PCA projection of the HSV feature space, the image labeled as glacier (red star) appears among neighbors consistently labeled as forest (green markers). Figure 10 shows the HSV histogram for this image. This illustrates our procedure of identifying images the algorithm marks as potentially miscategorized and manually inspecting them to verify their labels. 

<p align="center">
  <img src="task1/results/knn_suspicious/HSV_PCA.png" width="500"/><br>
  <em>Figure 9: PCA projection of HSV features highlighting a mislabeled glacier image</em>
</p>

<p align="center">
  <img src="task1/results/knn_suspicious/glacier_image_histogram.png" width="500"/><br>
  <em>Figure 10: HSV histogram for the mislabeled glacier image</em>
</p>

The neighbors used to detect the mislabeled image in Figure 10 and their HSV histograms are shown in Figure 11. These are joint 3D HSV histograms, where each bin counts pixels for a specific combination of hue, saturation, and value; once flattened, the x-axis therefore indexes bins rather than separate hue, saturation, or value channels. The forest images exhibit strong peaks around bin index 100, indicating many pixels share similar greenish HSV values characteristic of dense foliage. The HSV histogram of the miscategorized image in Figure 10 closely matches this distribution, which explains why the forest images in Figure 11 were selected as its nearest neighbors.

<p align="center">
  <img src="task1/results/knn_suspicious/forest_neighbor_histograms.png" width="500"/><br>
  <em>Figure 11: HSV histograms of forest images</em>
</p>

Table 3 shows how many potentially miscategorized images the algorithm detected in each category. Figures 12–17 show, for each category, the 40 images whose labels disagree most with their neighbors according to the KNN-based neigboor difference threshold explained earlier. This means that those images are most likely to differ from their assigned class label. These are the images most likely to be mislabeled relative to their assigned class. Despite KNN flagging 149 suspicious images in the mountains category, closer inspection revealed an even larger number of truly miscategorized images in the glacier class, as shown in Figure 12. We therefore manually inspected the glacier folder and removed 63 images from this category. 

We could have experimented with other parameter setting to detect more suspicious images, but we chose not to, since this method was only intended as a tool to highlight candidates for manual inspection. Our goal was to identify which types of miscategorized images appeared most frequently, not to develop a fully optimized automatic detector, so further tuning of the algorithm would have been unnecessarily demanding. The method suggested that miscategorized images were most prevalent in the glacier category.

In total, we removed 63 miscategorized images from the glacier category, most of which were images of flowers, animals, forest scenes, indoor environments, or lakes, and some are shown in figure 12. We acknowledge that some noisy or mislabeled data may still remain in the dataset, but we consider this acceptable given that the KNN results suggest only a small number of additional suspicious cases in the other categories, and that further cleaning would require substantial manual effort. We also chose not to reassign the removed glacier images to other categories, since many did not clearly belong to any of the predefined classes and they represent only a very small fraction of the overall dataset.

<div align="center">

| **Class**     | **Total images** | **Suspicious images** | **Share suspicious** |
|---------------|------------------|------------------------|---------------------|
| **Buildings** | 2 191            | 103                    | 4.70%               |
| **Forest**    | 2 271            | 15                     | 0.66%               |
| **Glacier**   | 2 404            | 48                     | 2.00%               |
| **Mountain**  | 2 512            | 149                    | 5.93%               |
| **Sea**       | 2 274            | 112                    | 4.93%               |
| **Street**    | 2 382            | 60                     | 2.52%               |
| **Total**     | 14 034           | 537                    | 3.5%                |


</div>

<p align="center"><em>Table 3: Potentially miscategorized images detected by K-Nearest-Neighboor.</em></p>

<p align="center">
  <img src="task1/results/knn_suspicious/buildings_suspicious.png" width="500"/><br>
  <em>Figure 12: Potentially miscategorized images in buildings category</em>
</p>

<p align="center">
  <img src="task1/results/knn_suspicious/forest_suspicious.png" width="500"/><br>
  <em>Figure 13: Potentially miscategorized images in forest category
  </em>
</p>

<p align="center">
  <img src="task1/results/knn_suspicious/glacier_suspicious.png" width="500"/><br>
  <em>Figure 14: Potentially miscategorized images in glacier category</em>
</p>

<p align="center">
  <img src="task1/results/knn_suspicious/mountain_suspicious.png" width="500"/><br>
  <em>Figure 15: Potentially miscategorized images in mountain category</em>
</p>

<p align="center">
  <img src="task1/results/knn_suspicious/sea_suspicious.png" width="500"/><br>
  <em>Figure 16: Potentially miscategorized images in sea category</em>
</p>

<p align="center">
  <img src="task1/results/knn_suspicious/street_suspicious.png" width="500"/><br>
  <em>Figure 17: Potentially miscategorized images in street category</em>
</p>

**Data Augmentation**

To increase robustness and expand the effective training set, we applied three forms of data augmentation to the training images: horizontal flipping, affine skewing, and central cropping followed by resizing. These augmentations are particularly relevant for our scene-classification task involving the categories buildings, forest, glacier, mountain, sea, and street, since natural scene images often appear with variations in viewpoint, framing, and orientation. Some examples of augmented images are shown in figure 18.

- **Horizontal flipping** helps the model become invariant to left–right orientation, which is common in landscapes and urban environments (e.g., coastlines, tree lines, or streets viewed from opposite angles).
- **Affine skewing** simulates changes in camera angle or perspective, which frequently occur in scenes such as buildings, mountains, and forests where the viewer's position can vary widely.
- **Central cropping** encourages the model to remain robust to shifts in zoom or framing, reflecting real-world variation in how scenes are captured.

<p align="center">
  <img src="task1/results/data_augmentation_examples.png" width="500"/><br>
  <em>Figure 18: Data augmented images</em>
</p>

**Resizing Images**

All images were originally 150×150 pixels but were resized to 128×128 during preprocessing. A fixed input size is required for the CNN and greatly simplifies feature extraction for the Random Forest, XGBoost, and stacking ensemble models. The 128×128 resolution offers a good balance between preserving visual detail and keeping the computational cost low, enabling efficient training while maintaining sufficient image quality for all models. Figure 19 shows that the clear structure and overall quality of each image are preserved at this lower resolution, indicating that 128×128 is sufficient for subsequent modelling.

<p align="center">
  <img src="task1/results/before_after_resizing.png" width="500"/><br>
  <em>Figure 19: Images before and after resizing.</em>
</p>

**Normalizing Images**

After resizing, we normalized all images by scaling pixel values from the original 0–255 range to 0–1. This keeps the overall pixel distribution intact as shown in figure 20 but puts all inputs on a common scale, which stabilizes training for the CNN and makes the features more comparable for the Random Forest, XGBoost, and stacking ensemble models.

<p align="center">
  <img src="task1/results/pixel_normalization_hist.png" width="500"/><br>
  <em>Figure 20: Images before and after resizing.</em>
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
