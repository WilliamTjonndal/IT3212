# IT3212 Assignment 2: Image Preprocessing

## Table of Contents

- [Fourier Transformation](#1-fourier-transformation)
  - [1. Load a grayscale image and apply the 2D Discrete Fourier Transform (DFT) to it Visualize the original image and its frequency spectrum (magnitude). Submit the images, and explanation.](#DFT-section-1)
  - [2. Implement a low-pass filter in the frequency domain to remove high-frequency noise from an image. Compare the filtered image with the original image. Submit images, and analysis of the results](#DFT-section-2)
  - [3. Implement a high-pass filter to enhance the edges in an image Visualize the filtered image and discuss the effects observed. Submit images, and explanation.](#DFT-section-3)
  - [4. Implement an image compression technique using Fourier Transform by selectively keeping only a certain percentage of the Fourier coefficients. Evaluate the quality of the reconstructed image as you vary the percentage of coefficients used. Submit the images,and your observations on image quality and compression ratio.](#DFT-section-4)

- [Principal Component Analysis](#2-principal-component-analysis)
    - [1. PCA Implementation.](#PCA-section-1)
    - [2. Reconstruction of images.](#PCA-section-2)
      - [a. Using the selected principal components, reconstruct the images.](#PCA-section-2a)
      - [b. Compare the reconstructed images with the original images to observe th effects of dimensionality reduction.](#PCA-section-2b)
    - [3. Experementation.](#PCA-section-3)
      - [a. Vary the number of principal components (k) and observe the impact on the quality of the reconstructed images.](#PCA-section-3a)
      - [b .Plot the variance explained by the principal components and determine the optimal number of components that balances compression and quality.](#PCA-section-3b)
    - [4. Visual Analysis.](#PCA-section-4)
      - [a. Display the original images alongside the reconstructed images for different values of k.](#PCA-section-4a)
      - [b. Comment on the visual quality of the images and how much information is lost during compression.](#PCA-section-4b)
    - [5. Error Analysis.](#PCA-section-5)
      - [a. Compute the Mean Squared Error (MSE) between the original and reconstructed images.](#PCA-section-5a)
      - [b. Analyze the trade-off between compression and reconstruction error.](#PCA-section-5b)

- [Local Binary Patterns](#3-local-binary-patterns)

- [Implement a Blob Detection Algorithm](#4-blob-detection)
   - [1. Apply the blob detection algorithm to one of the provided image datasets on blackborad.](#blob-section-1)
   - [2. Visualize the detected blobs on the original images, marking each detected blob with a circle or bounding box.](#blob-section-1)
   - [3. Calculate and display relevant statistics for each image, such as the number of blobs detected, their sizes, and positions.](#blob-section-3)

- [Implement a Contour Detection Algorithm](#5-contour-detection)
  - [1. Apply the contour detection algorithm to the same image dataset.](#contour-section-1)
  - [2. Visualize the detected contours on the original images, marking each contour with a different color.](#contour-section-2)
  - [3. Calculate and display relevant statistics for each image, such as the number of contours detected, contour area, and perimeter.](#contour-section-3)
  - [4. Compare the results of blob detection and contour detection for the chosen dataset.](#contour-section-4)
  - [5. Discuss the advantages and limitations of each technique.](#contour-section-5)
  - [6. Analyze the impact of different parameters (e.g., threshold values, filter sizes) on the detection results.](#contour-section-6)
  - [7. Provide examples where one technique might be more suitable than the other.](#contour-section-7)

<div style="page-break-after: always;"></div>









## <a id="1-fourier-transformation"></a> Fourier Transformation

### <a id="DFT-section-1"></a> 1. Load a grayscale image and apply the 2D Discrete Fourier Transform (DFT) to it Visualize the original image and its frequency spectrum (magnitude). Submit the images, and explanation.

<p align="center">
  <img src="results/dft/dft.png" width="500"/><br>
  <em>Figure 1: Discrete fourier transformation</em>
</p>

The 2D Discrete Fourier Transform (DFT) is used in image processing to convert an image from the spatial domain to the frequency domain. In this domain, each point represents a specific frequency and orientation, with the center representing the lowest frequencies (average intensity) and the outer points representing higher frequencies (detail and sharp changes). An inverse DFT transforms the frequency representation back into the spatial domain, and reconstruct the image. This technique is used for tasks like image filtering, compression and noice reduction by filtering high- and low frequencies.

### <a id="DFT-section-2"></a> 2. Implement a low-pass filter in the frequency domain to remove high-frequency noise from an image. Compare the filtered image with the original image. Submit images, and analysis of the results

A low-pass filter in the DFT keeps the low-frequency components near the spectrum’s center and suppresses high-frequency components toward the edges. After applying the inverse DFT, the loss of high-frequency detail like edges and fine textures produces a blurred image, as seen in Figure 2.

<p align="center">
  <img src="results/dft/lpf.png" width="500"/><br>
  <em>Figure 2: Low-pass filter</em>
</p>

### <a id="DFT-section-3"></a> 3. Implement a high-pass filter to enhance the edges in an image. Visualize the filtered image and discuss the effects observed. Submit images, and explanation.

A DFT high-pass filter preserves the high-frequency components toward the spectrum’s edges while suppressing the low-frequency components near the center. After the inverse DFT, the retained high-frequency detail emphasizes edges and fine textures, yielding a crisper, more contrasty result, as shown in Figure 3.

<p align="center">
  <img src="results/dft/hpf.png" width="500"/><br>
  <em>Figure 3: High-pass filter</em>
</p>

### <d id="DFT-section-4"></a> 4. Implement an image compression technique using Fourier Transform by selectively keeping only a certain percentage of the Fourier coefficients. Evaluate the quality of the reconstructed image as you vary the percentage of coefficients used. Submit the images, and your observations on image quality and compression ratio.

<p align="center">
  <img src="results/dft/coefficients.png" width="500"/><br>
  <em>Figure 4: Discrete fourier transformation coeffisients</em>
</p>

Keeping only a percentage of the Fourier coefficients means ranking the DFT coefficients by magnitude and retaining just the largest oneswhile zeroing the rest. As the proportion of retained Fourier coefficients increases, the reconstructed image quality improves gradually. 

When keeping only 1–5% of the coefficients, the image remains recognizable but appears blurred and lacks detail. At 10–20%, most structures and textures are restored, showing a noticeable enhancement in visual quality. From 30–50%, the reconstruction becomes nearly indistinguishable from the original, with PSNR values exceeding 40 dB. These results demonstrate that the essential visual information is concentrated in a small fraction of large-magnitude Fourier coefficients, enabling strong compression with minimal perceptual loss — effectively balancing compression ratio and image fidelity.

<div style="page-break-after: always;"></div>











## <a id="2-principal-component-analysis"></a> Principal Component Analysis

### <a id="PCA-section-1"></a>1. PCA Implementation

### <a id="PCA-section-2"></a>2. Reconstruction of images

#### <a id="PCA-section-2a"></a>a. Using the selected principal components, reconstruct the images.

#### <a id="PCA-section-2b"></a>b. Compare the reconstructed images with the original images to observe the effects of dimensionality reduction.

### <a id="PCA-section-3"></a>3. Experimentation

#### <a id="PCA-section-3a"></a>a. Vary the number of principal components (k) and observe the impact on the quality of the reconstructed images.

#### <a id="PCA-section-3b"></a>b. Plot the variance explained by the principal components and determine the optimal number of components that balances compression and quality.

### <a id="PCA-section-4"></a>4. Visual Analysis

#### <a id="PCA-section-4a"></a>a. Display the original images alongside the reconstructed images for different values of k.

#### <a id="PCA-section-4b"></a>b. Comment on the visual quality of the images and how much information is lost during compression.

### <a id="PCA-section-5"></a>5. Error Analysis

#### <a id="PCA-section-5a"></a>a. Compute the Mean Squared Error (MSE) between the original and reconstructed images.

#### <a id="PCA-section-5b"></a>b. Analyze the trade-off between compression and reconstruction error.

<div style="page-break-after: always;"></div>











## <a id="4-blob-detection"></a> Implement a Blob Detection Algorithm

### <a id="blob-section-1"></a>1. Apply the blob detection algorithm to one of the provided image datasets on blackboard.

Original images for reference:

<p align="center">
  <img src="img/blob/24212.jpg" width="300"/><br>
  <em>Image 1</em>
</p>

<p align="center">
</p>

<p align="center">
  <img src="img/blob/24230.jpg" width="300"/><br>
  <em>Image 3</em>
</p>

<p align="center">
  <img src="img/blob/24231.jpg" width="300"/><br>
  <em>Image 4</em>
</p>

<p align="center">
  <img src="img/blob/24250.jpg" width="300"/><br>
  <em>Image 5</em>
</p>

Below are the same images but with the Laplacian of Gaussian (LoG) Blob Detection algorithm applied on the grayscale version of the images.

<p align="center">
  <img src="results/blob/blob1-grayscale.png" width="300"/><br>
  <em>Figure 10a: Blob detection on image 1</em>
</p>

<p align="center">
  <img src="results/blob/blob2-grayscale.png" width="300"/><br>
  <em>Figure 10b: Blob detection on image 2</em>
</p>

<p align="center">
  <img src="results/blob/blob3-grayscale.png" width="300"/><br>
  <em>Figure 10c: Blob detection on image 3</em>
</p>

<p align="center">
  <img src="results/blob/blob4-grayscale.png" width="300"/><br>
  <em>Figure 10d: Blob detection on image 4</em>
</p>

<p align="center">
  <img src="results/blob/blob5-grayscale.png" width="300"/><br>
  <em>Figure 10e: Blob detection on image 5</em>
</p>

### <a id="blob-section-2"></a>2. Visualize the detected blobs on the original images, marking each detected blob with a circle or bounding box.

<p align="center">
  <img src="results/blob/blob1-.png" width="300"/><br>
  <em>Figure 11a: Blob detection on image 1</em>
</p>

<p align="center">
  <img src="results/blob/blob2-.png" width="300"/><br>
  <em>Figure 11b: Blob detection on image 2</em>
</p>

<p align="center">
  <img src="results/blob/blob3-.png" width="300"/><br>
  <em>Figure 11c: Blob detection on image 3</em>
</p>

<p align="center">
  <img src="results/blob/blob4-.png" width="300"/><br>
  <em>Figure 11d: Blob detection on image 4</em>
</p>

<p align="center">
  <img src="results/blob/blob5-.png" width="300"/><br>
  <em>Figure 11e: Blob detection on image 5</em>
</p>

### <a id="blob-section-3"></a>3. Calculate and display relevant statistics for each image, such as the number of blobs detected, their sizes, and positions.

<p align="center">
  <img src="results/blob/blob_analysis1-.png" width="800"/><br>
  <em>Figure 12a: Statistics for blob detection on image 1</em>
</p>

<p align="center">
  <img src="results/blob/blob_analysis2-.png" width="800"/><br>
  <em>Figure 12b: Statistics for blob detection on image 2</em>
</p>

<p align="center">
  <img src="results/blob/blob_analysis3-.png" width="800"/><br>
  <em>Figure 12c: Statistics for blob detection on image 3</em>
</p>

<p align="center">
  <img src="results/blob/blob_analysis4-.png" width="800"/><br>
  <em>Figure 12d: Statistics for blob detection on image 4</em>
</p>

<p align="center">
  <img src="results/blob/blob_analysis5-.png" width="800"/><br>
  <em>Figure 12e: Statistics for blob detection on image 5</em>
</p>

### <a id="blob-section-3"></a>4. Evaluate and discuss the effect of different parameters in the algorithms on the detection of different blobs.

As mentioned above, our blob detection algorithm uses the Laplacian of Gaussian (LoG) method from skimage.feature.blob_log.

The `max_sigma` parameter defines the maximum standard deviation for the Gaussian kernel and essentially sets the upper limit for the size of blobs that can be detected. We have set this to 30, which allows detection of relatively large blobs.\
If `max_sigma` is set too low, larger blobs will not be detected at all. On the other hand, a high value can lead to the detection of large, low-contrast regions that may not correspond to meaningful features.

The `num_sigma` parameter defines how many intermediate scales are tested between $0$ and `max_sigma`. We set a value value of $10$ so our code checks 10 different scales. Increasing this number can improve the precision of blob detection, especially for blobs that do not fall neatly into one of the predefined scales. However this can also greatly increases computational complexity.

The `threshold` parameter determines the minimum intensity difference required for a region to be considered a blob. A low `threshold` like $0.05$ makes the algorithm more sensitive, allowing it to detect faint or low-contrast blobs, but it may also detect noise.\
Conversely, a high `threshold` like $0.2$ makes the detection stricter, potentially missing subtle features while reducing false positives.

The overlay of detected blobs on grayscale and RGB images helps confirm whether the blobs align with visually identifiable features or not.\
Histograms of blob sizes reveal the distribution of detected radii across images and can indicate whether certain sizes are being over- or under-represented.\
The 2D heatmaps of blob positions show where blobs tend to occur spatially, revealing patterns or clustering, and can also highlight issues such as biased detection in bright regions due to thresholding.

<div style="page-break-after: always;"></div>







## <a id="5-contour-detection"></a> Implement a Contour Detection Algorithm

### <a id="contour-section-1"></a> 1. Apply the contour detection algorithm to the same image dataset.


Below are the same images but with the Marching Squares contour detection algorithm (skimage.measure.find_contours) applied on the grayscale version of the images.

<p align="center">
  <img src="results/contour/contour1.png" width="300"/><br>
  <em>Figure 13a: Contour detection on image 1</em>
</p>

<p align="center">
  <img src="results/contour/contour2.png" width="300"/><br>
  <em>Figure 13b: Contour detection on image 2</em>
</p>

<p align="center">
  <img src="results/contour/contour3.png" width="300"/><br>
  <em>Figure 13c: Contour detection on image 3</em>
</p>

<p align="center">
  <img src="results/contour/contour4.png" width="300"/><br>
  <em>Figure 13d: Contour detection on image 4</em>
</p>

<p align="center">
  <img src="results/contour/contour5.png" width="300"/><br>
  <em>Figure 13e: Contour detection on image 5</em>
</p>

### <a id="contour-section-2"></a> 2. Visualize the detected contours on the original images, marking each contour with a different color.

<p align="center">
  <img src="results/contour/contour1c.png" width="300"/><br>
  <em>Figure 14a: Contour detection on image 1</em>
</p>

<p align="center">
  <img src="results/contour/contour2c.png" width="300"/><br>
  <em>Figure 14b: Contour detection on image 2</em>
</p>

<p align="center">
  <img src="results/contour/contour3c.png" width="300"/><br>
  <em>Figure 14c: Contour detection on image 3</em>
</p>

<p align="center">
  <img src="results/contour/contour4c.png" width="300"/><br>
  <em>Figure 14d: Contour detection on image 4</em>
</p>

<p align="center">
  <img src="results/contour/contour5c.png" width="300"/><br>
  <em>Figure 14e: Contour detection on image 5</em>
</p>

### <a id="contour-section-3"></a> 3. Calculate and display relevant statistics for each image, such as the number of contours detected, contour area, and perimeter.

<p align="center">
  <img src="results/contour/results1.png" width="300"/>
  <img src="results/contour/contour1l.png" width="300"/>
  <br>
  <em>Figure 15a: Statistics for contour detection on image 1</em>
</p>

<p align="center">
  <img src="results/contour/results2.png" width="300"/>
  <img src="results/contour/contour2l.png" width="300"/><br>
  <em>Figure 15b: Statistics for contour detection on image 2</em>
</p>

<p align="center">
  <img src="results/contour/results3.png" width="300"/>
  <img src="results/contour/contour3l.png" width="300"/><br>
  <em>Figure 15c: Statistics for contour detection on image 3</em>
</p>

<p align="center">
  <img src="results/contour/results4.png" width="300"/>
  <img src="results/contour/contour4l.png" width="300"/><br>
  <em>Figure 15d: Statistics for contour detection on image 4</em>
</p>

<p align="center">
  <img src="results/contour/results5.png" width="300"/>
  <img src="results/contour/contour5l.png" width="300"/><br>
  <em>Figure 15e: Statistics for contour detection on image 5</em>
</p>

<!-- HISTOGRAMS -->

<p align="center">
  <img src="results/contour/contour1h.png" width="700"/><br>
  <em>Figure 15f: Histogram of statistics for contour detection on image 1</em>
</p>

<p align="center">
  <img src="results/contour/contour2h.png" width="700"/><br>
  <em>Figure 15g: Histogram of statistics for contour detection on image 2</em>
</p>

<p align="center">
  <img src="results/contour/contour3h.png" width="700"/><br>
  <em>Figure 15i: Histogram of statistics for contour detection on image 3</em>
</p>

<p align="center">
  <img src="results/contour/contour4h.png" width="700"/><br>
  <em>Figure 15j: Histogram of statistics for contour detection on image 4</em>
</p>

<p align="center">
  <img src="results/contour/contour5h.png" width="700"/><br>
  <em>Figure 15k: Histogram of statistics for contour detection on image 5</em>
</p>

### <a id="contour-section-4"></a> 4. Compare the results of blob detection and contour detection for the chosen dataset.

<p align="center">
  <img src="results/blobvscontour1.png" width="300"/><br>
  <em>Figure 16a: Blob vs contour detection on image 1</em>
</p>

<p align="center">
  <img src="results/blobvscontour2.png" width="300"/><br>
  <em>Figure 16b: Blob vs contour detection on image 2</em>
</p>

<p align="center">
  <img src="results/blobvscontour3.png" width="300"/><br>
  <em>Figure 16c: Blob vs contour detection on image 3</em>
</p>

<p align="center">
  <img src="results/blobvscontour4.png" width="300"/><br>
  <em>Figure 16d: Blob vs contour detection on image 4</em>
</p>

<p align="center">
  <img src="results/blobvscontour5.png" width="300"/><br>
  <em>Figure 16e: Blob vs contour detection on image 5</em>
</p>

### <a id="contour-section-5"></a> 5. Discuss the advantages and limitations of each technique.

Blob detection is efficient at identifying roughly circular regions and provides quick localization and size estimates, making it ideal for detecting spots or particles across multiple scales.\
However, it lacks detailed shape information and struggles with irregular or complex objects.\
Contour detection, on the other hand, excels at outlining precise object boundaries and capturing detailed shape features, which is valuable for morphological analysis.\
Its effectiveness depends heavily on image quality and edge definition, and it can be computationally more intensive and sensitive to noise.\
So blob detection is best for fast, approximate feature localization, while contour detection is preferred when detailed shape and boundary information is required.

### <a id="contour-section-6"></a> 6. Analyze the impact of different parameters (e.g., threshold values, filter sizes) on the detection results.

<p align="center">
  <img src="results/blob_analysis.png" width="800"/><br>
  <em>Figure 17a: Histogram of statistics for contour detection on image 1</em>
</p>

<p align="center">
  <img src="results/contour_analysis.png" width="800"/><br>
  <em>Figure 17b: Histogram of statistics for contour detection on image 1</em>
</p>

The performance of both blob and contour detection methods is sensitive to parameters like threshold values and filter sizes.\
In blob detection, adjusting the `threshold` controls the sensitivity.\
Lower thresholds detect more blobs but increase false positives, while higher thresholds reduce noise but may miss subtle features. Similarly, the choice of `max_sigma` and `num_sigma` affects the scale range and granularity of detected blobs.\
For contour detection, the threshold used in binarization critically impacts which features are segmented, a too low threshold may merge objects or include noise, while too high may fragment or miss contours.\
Additionally, morphological operations like removing small objects depend on filter sizes that balance noise reduction against losing small meaningful contours.

### <a id="contour-section-7"></a> 7. Provide examples where one technique might be more suitable than the other.

Blob detection is more suitable in applications where the target objects are roughly circular and uniformly bright or dark against the background.\
For example, detecting cells in microscopy images, stars in astronomical images, or bubbles in fluid simulations.\
Its strength lies in quick localization and size estimation of round features across different scales.\
In contrast, contour detection is ideal when precise object boundaries and shape details are essential, such as in medical image analysis, character recognition, or analyzing irregularly shaped objects like leaves or cracks.\
It enables detailed morphological analysis, making it preferable when shape complexity and boundary accuracy matter more than speed or rough position.




