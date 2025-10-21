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

### <a id="blob-section-1"></a> Apply the blob detection algorithm to one of the provided image datasets on blackboard.

### <a id="blob-section-2"></a> Visualize the detected blobs on the original images, marking each detected blob with a circle or bounding box.

<p align="center">
  <img src="results/blob/blob1.png" width="500"/><br>
  <em>Figure 3a Blob detection on image 1</em>
</p>

<p align="center">
  <img src="results/blob/blob2.png" width="500"/><br>
  <em>Figure 3b Blob detection on image 2</em>
</p>

<p align="center">
  <img src="results/blob/blob3.png" width="500"/><br>
  <em>Figure 3c Blob detection on image 3</em>
</p>

<p align="center">
  <img src="results/blob/blob4.png" width="500"/><br>
  <em>Figure 3d Blob detection on image 4</em>
</p>

### <a id="blob-section-3"></a> 3. Calculate and display relevant statistics for each image, such as the number of blobs detected, their sizes, and positions.

<p align="center">
  <img src="results/blob/blob_analysis1.png" width="500"/><br>
  <em>Figure 3a Blob detection on image 1</em>
</p>

<p align="center">
  <img src="results/blob/blob_analysis2.png" width="500"/><br>
  <em>Figure 3b Blob detection on image 2</em>
</p>

<p align="center">
  <img src="results/blob/blob_analysis3.png" width="500"/><br>
  <em>Figure 3c Blob detection on image 3</em>
</p>

<p align="center">
  <img src="results/blob/blob_analysis4.png" width="500"/><br>
  <em>Figure 3d Blob detection on image 4</em>
</p>

### <a id="blob-section-3"></a> Evaluate and discuss the effect of different parameters in the algorithms on the detection of different blobs.

The blob detection algorithm implemented in your code uses the Laplacian of Gaussian (LoG) method from skimage.feature.blob_log, and the effectiveness of the detection is heavily influenced by the parameters passed to this function. The key parameters—max_sigma, num_sigma, and threshold—control how the algorithm identifies and responds to features in the image.

The max_sigma parameter defines the maximum standard deviation for the Gaussian kernel and essentially sets the upper limit for the size of blobs that can be detected. In your code, this is set to 30, which allows detection of relatively large blobs. If max_sigma is set too low, larger blobs will not be detected at all. On the other hand, a very high value can lead to the detection of large, low-contrast regions that may not correspond to meaningful features, increasing the likelihood of false positives. Adjusting this parameter allows you to target blobs of different sizes depending on the nature of your images.

The num_sigma parameter defines how many intermediate scales are tested between 0 and max_sigma. With a value of 10 in your code, the algorithm checks 10 different scales. Increasing this number can improve the precision of blob detection, especially for blobs that do not fall neatly into one of the predefined scales. However, this also increases computational complexity. Fewer values speed up processing but can miss blobs whose sizes lie between the sampled scales.

The threshold parameter determines the minimum intensity difference required for a region to be considered a blob. A low threshold (e.g., 0.05) makes the algorithm more sensitive, allowing it to detect faint or low-contrast blobs, but it may also detect noise. Conversely, a high threshold (e.g., 0.2) makes the detection stricter, potentially missing subtle features while reducing false positives. Your current value of 0.1 represents a balanced starting point, though tuning it based on the contrast and quality of your images may yield better results.

Visualizations in your code play a crucial role in evaluating the impact of these parameters. The overlay of detected blobs on grayscale and RGB images helps confirm whether the blobs align with visually identifiable features or not. Histograms of blob sizes reveal the distribution of detected radii across images and can indicate whether certain sizes are being over- or under-represented, possibly suggesting adjustments to max_sigma. The 2D heatmaps of blob positions show where blobs tend to occur spatially, revealing patterns or clustering, and can also highlight issues such as biased detection in bright regions due to thresholding.

In summary, the choice of max_sigma, num_sigma, and threshold significantly affects blob detection outcomes. Fine-tuning these parameters based on your specific image set—considering blob size, image contrast, and noise levels—is essential for optimal performance. Using diagnostic visualizations like size histograms and heatmaps can guide effective parameter adjustments.

<div style="page-break-after: always;"></div>











## <a id="5-contour-detection"></a> Implement a Contour Detection Algorithm

### <a id="contour-section-1"></a> 1. Apply the contour detection algorithm to the same image dataset.

### <a id="contour-section-2"></a> 2. Visualize the detected contours on the original images, marking each contour with a different color.

### <a id="contour-section-3"></a> 3. Calculate and display relevant statistics for each image, such as the number of contours detected, contour area, and perimeter.

### <a id="contour-section-4"></a> 4. Compare the results of blob detection and contour detection for the chosen dataset.

### <a id="contour-section-5"></a> 5. Discuss the advantages and limitations of each technique.

### <a id="contour-section-6"></a> 6. Analyze the impact of different parameters (e.g., threshold values, filter sizes) on the detection results.

### <a id="contour-section-7"></a> 7. Provide examples where one technique might be more suitable than the other.





