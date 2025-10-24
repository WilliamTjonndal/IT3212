# IT3212 Assignment 2: Image Preprocessing

## Table of Contents

  - [Fourier Transformation](#1-fourier-transformation)
    - [ 1. Load a grayscale image and apply the 2D Discrete Fourier Transform (DFT) to it Visualize the original image and its frequency spectrum (magnitude). Submit the images, and explanation.](#DFT-section-1)
    - [ 2. Implement a low-pass filter in the frequency domain to remove high-frequency noise from an image. Compare the filtered image with the original image. Submit images, and analysis of the results](#DFT-section-2)
    - [ 3. Implement a high-pass filter to enhance the edges in an image. Visualize the filtered image and discuss the effects observed. Submit images, and explanation.](#DFT-section-3)
    - [ 4. Implement an image compression technique using Fourier Transform by selectively keeping only a certain percentage of the Fourier coefficients. Evaluate the quality of the reconstructed image as you vary the percentage of coefficients used. Submit the images, and your observations on image quality and compression ratio.](#DFT-section-4)

- [PCA](#2-principal-component-analysis)
  - [1. PCA Implementation](#PCA-section-1)
  - [2. Reconstruction of images](#PCA-section-2)
    - [a. Using the selected principal components, reconstruct the images.](#PCA-section-2a)
    - [b. Compare the reconstructed images with the original images to observe the effects of dimensionality reduction.](#PCA-section-2b)
  - [3. Experimentation](#PCA-section-3)
    - [a. Vary the number of principal components (k) and observe the impact on the quality of the reconstructed images.](#PCA-section-3a)
    - [b. Plot the variance explained by the principal components and determine the optimal number of components that balances compression and quality.](#PCA-section-3b)
  - [4. Visual Analysis](#PCA-section-4)
    - [a. Display the original images alongside the reconstructed images for different values of k.](#PCA-section-4a)
    - [b. Comment on the visual quality of the images and how much information is lost during compression.](#PCA-section-4b)
  - [5. Error Analysis](#PCA-section-5)
    - [a. Compute the Mean Squared Error (MSE) between the original and reconstructed images.](#PCA-section-5a)
    - [b. Analyze the trade-off between compression and reconstruction error.](#PCA-section-5b)


- [Histogram of Oriented Gradients](#3-histogram-of-oriented-gradients)
    - [1. Write a Python script to compute the HOG features of a given image using a library such as OpenCV or scikit-image. Apply your implementation to at least three different images, including both simple and complex scenes.](#hog-section-1)
    - [2. Visualize the original image, the gradient image, and the HOG feature image. Compare the HOG features extracted from different images.](#hog-section-2)
    - [3. Discuss the impact of varying parameters like cell size, block size, and the number of bins on the resulting HOG descriptors.](#hog-section-3)

- [Local Binary Patterns](#4-local-binary-patterns)
    - [1. Write a Python function to compute the LBP of a given grayscale image (basic 8-neighbor). Your function should output the LBP image, where each pixel is replaced by its corresponding LBP value.](#lbp-section-1)
    - [2. Write a Python function to compute the histogram of the LBP image. Plot the histogram and explain what it represents in terms of the texture features of the image.](#lbp-section-2)
    - [3. Apply your LBP function to at least three different grayscale images (e.g., a natural scene, a texture, and a face image). Generate and compare the histograms of the LBP images.](#lbp-section-3)
    - [4. Discuss the differences in the histograms and what they tell you about the textures of the different images.](#lbp-section-4)

- [Implement a Blob Detection Algorithm.](#5-blob-detection)
    - [ 1. Apply the contour detection algorithm to the same image dataset. Visualize the detected contours on the original images, marking each contour with a different color.](#blob-section-1)
    - [ 2. Calculate and display relevant statistics for each image, such as the number of blobs detected, their sizes, and positions.](#blob-section-2)
    - [ 3. Evaluate and discuss the effect of different parameters in the algorithms on the detection of different blobs.](#blob-section-3)

- [Implement a Contour Detection Algorithm](#6-contour-detection)
    - [1. Apply the contour detection algorithm to the same image dataset. Visualize the detected contours on the original images, marking each contour with a different color.](#contour-section-1)
    - [2. Calculate and display relevant statistics for each image, such as the number of contours detected, contour area, and perimeter.](#contour-section-2)
    - [3. Compare the results of blob detection and contour detection for the chosen dataset.](#contour-section-3)
    - [ 4. Discuss the advantages and limitations of each technique.](#contour-section-4)
    - [5. Analyze the impact of different parameters (e.g., threshold values, filter sizes) on the detection results.](#contour-section-5)
    - [6. Provide examples where one technique might be more suitable than the other.](#contour-section-6)

<div style="page-break-after: always;"></div>

## <a id="1-fourier-transformation"></a> Fourier Transformation

### <a id="DFT-section-1"></a> 1. Load a grayscale image and apply the 2D Discrete Fourier Transform (DFT) to it Visualize the original image and its frequency spectrum (magnitude). Submit the images, and explanation.

<p align="center">
  <img src="results/dft/dft.png" width="500"/><br>
  <em>Figure 1: Discrete fourier transformation</em>
</p>

The 2D Discrete Fourier Transform (DFT) converts an image from the spatial domain to the frequency domain. In this domain, each point encodes a sinusoidal frequency and orientation, with the center representing the lowest frequencies (average intensity) and the outer points representing higher frequencies (detail and sharp changes). An inverse DFT transforms the frequency representation back into the spatial domain, and reconstruct the image. We see this in figure 1, the log-scale magnitude spectrum is brightest at the center (low frequencies) and sparse toward the edges (high frequencies), and the inverse DFT on the right reconstructs the image accordingly.

### <a id="DFT-section-2"></a> 2. Implement a low-pass filter in the frequency domain to remove high-frequency noise from an image. Compare the filtered image with the original image. Submit images, and analysis of the results

<p align="center">
  <img src="results/dft/lpf.png" width="500"/><br>
  <em>Figure 2: Low-pass filter</em>
</p>

A low-pass filter on the DFT keeps the low-frequency components near the spectrum’s center and suppresses high-frequency components toward the edges. After applying the inverse DFT, the loss of high-frequency detail like edges and fine textures produces a blurred image, as seen in figure 2.

### <a id="DFT-section-3"></a> 3. Implement a high-pass filter to enhance the edges in an image. Visualize the filtered image and discuss the effects observed. Submit images, and explanation.

<p align="center">
  <img src="results/dft/hpf.png" width="500"/><br>
  <em>Figure 3: High-pass filter</em>
</p>

A DFT high-pass filter preserves the high-frequency components toward the spectrum’s edges while suppressing the low-frequency components near the center. After the inverse DFT, the retained high-frequency detail emphasizes edges and fine textures, yielding a sharper image, as shown in figure 3.

### <d id="DFT-section-4"></a> 4. Implement an image compression technique using Fourier Transform by selectively keeping only a certain percentage of the Fourier coefficients. Evaluate the quality of the reconstructed image as you vary the percentage of coefficients used. Submit the images, and your observations on image quality and compression ratio.

<p align="center">
  <img src="results/dft/coefficients.png" width="500"/><br>
  <em>Figure 4: Discrete fourier transformation coeffisients</em>
</p>

Keeping only a percentage of the Fourier coefficients means ranking the DFT coefficients by magnitude and retaining just the largest ones, while zeroing the rest. As the proportion of retained Fourier coefficients increases, the visual quality of the reconstructed images improves gradually, as shown in figure 4. At 0.1% of coefficients, the image is barely recognizable, only the dog’s rough outline and overall composition are visible, with fine details lost. At 1-5%, most structures and textures are restored, but the reconstructions still remain blurry. From 5%, the reconstruction becomes nearly indistinguishable from the original image.

<p align="center">
  <img src="results/dft/CR.png" width="500"/><br>
  <em>Figure 5: Compression ratio</em>
</p>

Compression ratio (CR) measures how much an image is reduced in size, defined as compressed size divided by original size. A lower CR means more compression, and potentially greater quality loss, while a higher CR means less compression. As shown in figure 5, the compression ratio is linear, meaning that keeping more coefficients takes more space.


<p align="center">
  <img src="results/dft/PSNR.png" width="500"/><br>
  <em>Figure 6: Peak Signal-to-Noise Ratio</em>
</p>

Peak Signal-to-Noise Ratio (PSNR) computes the peak signal-to-noise ratio, in decibels, between two images. This ratio is used as a quality measurement between the original and a compressed image. The higher the PSNR, the better the quality of the compressed, or reconstructed image. In figure 6, PSNR rises quickly at very low keep rates, from about 20 dB at 0.1% to around 29–31 dB by 3–5%, then levels off, reaching roughly 37 dB at 20%. As noted in in figure 4, retaining more than 5% of coefficients yields little additional quality. This aligns with the PSNR curve in figure 6 where increases beyond 5% are marginal, and PSNR values above 30 dB already indicate good quality.

<p align="center">
  <img src="results/dft/SSIM.png" width="500"/><br>
  <em>Figure 7: Structural Similarity Index</em>
</p>

The Structural Similarity Index (SSIM) is a metric used to measure the similarity between two images by comparing luminance, contrast, and structure, where 1 indicates a perfect match. As shown in figure 7, SSIM rises sharply at very low keep-rates and then converges slowly to 1. Keeping more than 5% of the coeffisients only yield minor gains in similarity, which corresponds with the quality trends shown in figure 6.


<div style="page-break-after: always;"></div>











## <a id="2-principal-component-analysis"></a> Principal Component Analysis

### <a id="PCA-section-1"></a>1. PCA Implementation

<p align="center">
  <img src="results/pca/original.png" width="800"/><br>
  <em>Figure 8: Original images</em>
</p>

We loaded the original images in grayscale with a size of 128x128 pixels, then normalized all pixel values between 0 and 1.

The images were converted into a 2D matrix with image as row and pixel value as column.

A covariance matrix was calculated for the image matrix, and it was used to calculate eigenvalues and eigenvectors, sorted in descending order.

<p align="center">
  <img src="results/pca/components.png" width="800"/><br>
  <em>Figure 9: Principal Components</em>
</p>

We selected the top k = 6 eigenvectors to start with, and will experiment with the amount later. These were used to create our principal components, as seen in figure 9.

<p align="center">
  <img src="results/pca/2d.png" width="800"/><br>
  <em>Figure 10: Images visualized as PC1 vs PC2</em>
</p>

We visualized all 8 images in the 2-dimensional subspace defined by PC1 and PC2, shown in figure 10.

### <a id="PCA-section-2"></a>2. Reconstruction of images

#### <a id="PCA-section-2a"></a>a. Using the selected principal components, reconstruct the images.

<p align="center">
  <img src="results/pca/reconstructedwithk6.png" width="800"/><br>
  <em>Figure 11: Images reconstructed with k = 6</em>
</p>

#### <a id="PCA-section-2b"></a>b. Compare the reconstructed images with the original images to observe the effects of dimensionality reduction.
<p align="center">
  <img src="results/pca/originalvsk6.png" width="800"/><br>
  <em>Figure 12: Original images vs reconstructed images with k = 6</em>
</p>



### <a id="PCA-section-3"></a>3. Experimentation
#### <a id="PCA-section-3a"></a>a. Vary the number of principal components (k) and observe the impact on the quality of the reconstructed images.

<p align="center">
  <img src="results/pca/reconstructedwithdifferentKs.png" width="800"/><br>
  <em>Figure 13: Images reconstructed with different K's</em>
</p>

We experimented with different k values for the reconstruction of the original images. Figure 13 illustrates how the reconstruction increasingly aporaches an aproximation of the original images with every aditional increase of k.

#### <a id="PCA-section-3b"></a>b. Plot the variance explained by the principal components and determine the optimal number of components that balances compression and quality.

<p align="center">
  <img src="results/pca/varianceplot.png" width="800"/><br>
  <em>Figure 14: Plot for cumulative variance and individual variance per component </em>
</p>

With a threshold of 90% we see that six prinicple components would be needed to reach this level. As seen in figure 13 the images reconstructed with less than six components are considerably more blurry. Given that our dataset is a facial emotions dataset would mean that blurry images are detrimental to the intended purpose of the dataset. However, using all seven components would aproximate a full reconstruction of the original images and would constitute little compression. It can therefore be argued that in our case, if we want to compress our images we could only use six priniciple componets before it would make the subjects emotions difficult to recognize.

### <a id="PCA-section-4"></a>4. Visual Analysis

#### <a id="PCA-section-4a"></a>a. Display the original images alongside the reconstructed images for different values of k.

The original images alongside the reconstructed images for different values of k is shown if figure 13. 

#### <a id="PCA-section-4b"></a>b. Comment on the visual quality of the images and how much information is lost during compression.

As seen in figure 12, the reconstructed images are very similar to the originals, with the exeption of the fourth image from the left which is very blurry. You can still make out the expression in the image, but it's much harder than with the rest.

This problem can also be seen in figure 13, where reconstructions with lower values for k are more blurry than those with higher values.

You can also see a representation of the quality of the reconstruction using MSE in figure 15, and it will be discussed further in the next section.

### <a id="PCA-section-5"></a>5. Error Analysis

#### <a id="PCA-section-5a"></a>a. Compute the Mean Squared Error (MSE) between the original and reconstructed images.
<p align="center">
  <img src="results/pca/mse.png" width="800"/><br>
  <em>Figure 15: Orignal images comapred to reconstructed images with k = 6</em>
</p>

#### <a id="PCA-section-5b"></a>b. Analyze the trade-off between compression and reconstruction error.

<p align="center">
  <img src="results/pca/mseVScompression.png" width="800"/><br>
  <em>Figure 16: Orignal images comapred to reconstructed images with k = 6</em>
</p>

In figure 16, we can see that the MSE is continually decreasing from 0.028 (k = 1) to 0.0033 (k =6 ), and becomes aproximally 0 at k = 7, which is in line with the dataset being rank n - 1 = 7. We chose k = 6 as a balance between compression and quality, with a PNSR of about 25 dB.




<div style="page-break-after: always;"></div>

## <a id="3-histogram-of-oriented-gradients"></a> Histogram of Oriented Gradients

### <a id="hog-section-1"></a>1. Write a Python script to compute the HOG features of a given image using a library such as OpenCV or scikit-image. Apply your implementation to at least three different images, including both simple and complex scenes.

Histogram of Oriented Gradients (HOG) features capture local shape by counting how often edges point in each direction within small regions, then normalizing and concatenating them into a feature vector. The x and y gradients are the horizontal and vertical changes in pixel intensity, revealing edge direction. The gradient magnitude is the overall edge strength at each pixel, used as the weight when voting into orientation bins. HOG is a feature descriptor used for object detection.

### <a id="hog-section-2"></a>2. Visualize the original image, the gradient image, and the HOG feature image. Compare the HOG features extracted from different images.

<p align="center">
  <img src="results/hog/hog_features.png" width="800"/><br>
  <em>Figure 17: HOG features with baseline parameters</em>
</p>

As shown in figure 17, HOG is better at capturing sharp edges and overall shape/contour than fine textures. Consequently, it renders the human and car contour more clearly since the original images contain fewer details and has well-defined edges. The strawberry and tiger are harder to recognize from HOG features because their appearances are dominated by fine textures as seeds and fur. 

### <a id="hog-section-3"></a>3. Discuss the impact of varying parameters like cell size, block size, and the number of bins on the resulting HOG descriptors.

<p align="center">
  <img src="results/hog/hog_features_grid.png" width="800"/><br>
  <em>Figure 18: HOG features with different parameters</em>
</p>


As shown in figure 18, using smaller cells (4×4) makes the HOG more sensitive to fine textures and details. This is most visible on the image of the strawberry, where individual seeds are much more recognizable compared to HOG features with other parameters. Smaller cells also improves the tiger image, revealing more fine fur detail. Larger cells (16×16) smooth local gradients and emphasize only the rough shape of the image.

Block size has less impact than cell size, but tiny blocks (1×1) preserve more local contrast and are less robust to illumination/contrast changes, while large blocks (4×4 cells) normalize gradients across a wider area, improving robustness to illumination/contrast changes, but slightly smooths local variation.

The amount of orientation also does not have the same impact as cell size, but fewer orientation bins (6) give more compact, coarse angle coding that highlights major contours, while many bins (18) capture subtle angle changes but can add redundancy/noise.

<div style="page-break-after: always;"></div>



















## <a id="4-local-binary-patterns"></a> Local Binary Patterns

### <a id="lbp-section-1"></a>1. Write a Python function to compute the LBP of a given grayscale image (basic 8-neighbor). Your function should output the LBP image, where each pixel is replaced by its corresponding LBP value.

<p align="center">
  <img src="results/lbp/lbp.png" width="800"/><br>
  <em>Figure 19: LBP</em>
</p>

Local Binary Patterns (LBP) encodes local texture at each pixel by comparing the pixel's intensity to its eight immediate neighbors: 1 for each neighbor that is at least as bright as the center, otherwise 0. Reading these eight bits in a fixed order yields an 8-bit pattern that is converted to a decimal value in the range from 0 to 255, and the pixel in the LBP image is replaced by this value. Figure 19 presents the original image alongside its 8-neighbor LBP representation.

### <a id="lbp-section-2"></a>2. Write a Python function to compute the histogram of the LBP image. Plot the histogram and explain what it represents in terms of the texture features of the image.

<p align="center">
  <img src="results/lbp/lbp_histogram.png" width="800"/><br>
  <em>Figure 20: LBP histogram</em>
</p>

An LBP histogram counts how many pixels in the LBP image have each code value from 0 to 255, and shows the texture distribution within the original image. The histogram is used to capture the frequency of occurrence of different texture patterns in the original image.

In figure 20, the LBP histogram shows tall peaks near the extreme codes (0 and 255), indicating many uniform patterns.



### <a id="lbp-section-3"></a>3. Apply your LBP function to at least three different grayscale images (e.g., a natural scene, a texture, and a face image). Generate and compare the histograms of the LBP images.

<p align="center">
  <img src="results/lbp/lbp_grid.png" width="800"/><br>
  <em>Figure 21: LBP for several images</em>
</p>

The LBP histograms differs among the three images. The image of Mona Lisa's histogram has a more uniform distribution of pixel values, with smaller spikes and large spikes at 0 and 255.

The image of the bricks wall's histogram has a slightly less uniform distribuiton, with more sparse spikes.

The image of the landscape's histogram is much less uniformly distributed, with tall, sparse spikes and not much between them.

### <a id="lbp-section-4"></a>4. Discuss the differences in the histograms and what they tell you about the textures of the different images.

<p align="center">
  <img src="results/lbp/lbp_grid_category.png" width="800"/><br>
  <em>Figure 22: LBP by category for several images</em>
</p>

We categorized LBP pixel values into 4 categories, flat 0, edge (uniform), flat 255, and corner/noise. The categories are decided based on the binary encoding of the surrounding pixel values, where the amount of transitions between 0 and 1 decide the category.

No transitions in the binary pixel value, e.g. 00000000, means it's categoriezed as flat, going into either flat 0 or flat 255, depending on whether every number is 0 or 1.

One or two transitions in the binary pixel value, e.g. 11000111, means it's classified as an edge.

More than two transitions, e.g. 10101010, are classified as a corner/noise.

After visualizing the LBP results in this way, it's much easier to interpret our results. In the image of mona lisa, you can see the edges are clearly marked in red, while the rest of the image 
is filled with flat 0, flat 255, and corner/noise.

The image of the brick wall has much more noise, and it also has many corners, so it's filled with way more pixels classified as corner/noise. Not many of the brick edges are correctly classified as edges, which is because of choosing a radius of 4, which isn't as good at recognizing thin edges like on the bricks. The higher radius makes the LBP recognize thich edges easier, but is worse at detecting thin edges.

The image of the forest is also not classified as well as the image of mona lisa. This time, most of the sky is classified as edges, which is caused by it having a gradient, which is interpreted as an edge when using LBP.

We selected the parameters of the LBP function to improve the categorization for the image of mona lisa, which is why it's categories seem more accurate than for the brick wall or the landscape.

To receive better results, we would perform a discrete fourier transform on the images, reducing noise. We would also vary the parameters, the amount of neighbours and the radius, based on the dataset,\. This is so the LBP pixel value categories fit better for all images in the dataset instead of maximising quality for one of them.















## <a id="5-blob-detection"></a> Implement a Blob Detection Algorithm. 


### <a id="blob-section-1"></a> 1. Apply the blob detection algorithm to one of the provided image datasets on blackboard. Visualize the detected blobs on the original images, marking each detected blob with a circle or bounding box.

<p align="center">
  <img src="results/blob2/blob_detection.png" width="300"/><br>
  <em>Figure 23: Blob detection applied on greyscale images</em>
</p>

Blob detection algorithms identify regions in an image with distinct properties like brightness or color. Our implemented blob detection algorithm uses the Laplacian of Gaussian (LoG) method. The results are shown in figure 23.

### <a id="blob-section-2"></a> 2. Calculate and display relevant statistics for each image, such as the number of blobs detected, their sizes, and positions.

<p align="center">
  <img src="results/blob2/blob_detection_analysis.png" width="300"/><br>
  <em>Figure 24: Blob detection statistic</em>
</p>

The overlay of detected blobs on grayscale and RGB images helps confirm whether the blobs align with visually identifiable features or not.

Histograms of blob sizes reveal the distribution of detected radius across images and can indicate whether certain sizes are being over- or under-represented.

The 2D heatmaps of blob positions show where blobs tend to occur spatially, revealing patterns or clustering, and can also highlight issues such as biased detection in bright regions due to thresholding. 

### <a id="blob-section-3"></a> 3. Evaluate and discuss the effect of different parameters in the algorithms on the detection of different blobs.

The `max_sigma` parameter defines the maximum standard deviation for the Gaussian kernel and essentially sets the upper limit for the size of blobs that can be detected. We have set this to 30, which allows detection of relatively large blobs.

If `max_sigma` is set too low, larger blobs will not be detected at all. On the other hand, a high value can lead to the detection of large, low-contrast regions that may not correspond to meaningful features.

The `num_sigma` parameter defines how many intermediate scales are tested between $0$ and `max_sigma`. We set a value value of $10$ so our code checks 10 different scales. Increasing this number can improve the precision of blob detection, especially for blobs that do not fall neatly into one of the predefined scales. However this can also greatly increases computational complexity.

<p align="center">
  <img src="results/blob_analysis.png" width="800"/><br>
  <em>Figure 25: Blob detection with different thresholds</em>
</p>

The `threshold` parameter determines the minimum intensity difference required for a region to be considered a blob. A low `threshold` like $0.05$ makes the algorithm more sensitive, allowing it to detect faint or low-contrast blobs, but it may also detect noise. Conversely, a high `threshold` like $0.2$ makes the detection stricter, potentially missing subtle features while reducing false positives. This is shown in figure 25.

<div style="page-break-after: always;"></div>















## <a id="6-contour-detection"></a> Implement a Contour Detection Algorithm

### <a id="contour-section-1"></a> 1. Apply the contour detection algorithm to the same image dataset. Visualize the detected contours on the original images, marking each contour with a different color.

<p align="center">
  <img src="results/contour2/contour_detection.png" width="300"/><br>
  <em>Figure 26: Contour detection applied to greyscale images</em>
</p>

Contour detection algorithms aim to identify and extract the boundaries of objects within an image, often represented as a sequence of connected points or curves. Our implemented contour detection algorithm uses the Marching Squares  method. The results are shown in figure 26.

### <a id="contour-section-2"></a> 2. Calculate and display relevant statistics for each image, such as the number of contours detected, contour area, and perimeter.

<p align="center">
  <img src="results/contour/results1.png" width="300"/>
  <img src="results/contour/contour1l.png" width="300"/>
  <br>
  <em>Figure 27: Statistics for contour detection on image 1</em>
</p>

<p align="center">
  <img src="results/contour/results2.png" width="300"/>
  <img src="results/contour/contour2l.png" width="300"/><br>
  <em>Figure 28: Statistics for contour detection on image 2</em>
</p>

<p align="center">
  <img src="results/contour/results3.png" width="300"/>
  <img src="results/contour/contour3l.png" width="300"/><br>
  <em>Figure 29: Statistics for contour detection on image 3</em>
</p>

<p align="center">
  <img src="results/contour/results4.png" width="300"/>
  <img src="results/contour/contour4l.png" width="300"/><br>
  <em>Figure 30: Statistics for contour detection on image 4</em>
</p>

<p align="center">
  <img src="results/contour/results5.png" width="300"/>
  <img src="results/contour/contour5l.png" width="300"/><br>
  <em>Figure 31: Statistics for contour detection on image 5</em>
</p>

<!-- HISTOGRAMS -->

<p align="center">
  <img src="results/contour/contour1h.png" width="700"/><br>
  <em>Figure 32: Histogram of statistics for contour detection on image 1</em>
</p>

<p align="center">
  <img src="results/contour/contour2h.png" width="700"/><br>
  <em>Figure 33: Histogram of statistics for contour detection on image 2</em>
</p>

<p align="center">
  <img src="results/contour/contour3h.png" width="700"/><br>
  <em>Figure 34: Histogram of statistics for contour detection on image 3</em>
</p>

<p align="center">
  <img src="results/contour/contour4h.png" width="700"/><br>
  <em>Figure 35: Histogram of statistics for contour detection on image 4</em>
</p>

<p align="center">
  <img src="results/contour/contour5h.png" width="700"/><br>
  <em>Figure 36: Histogram of statistics for contour detection on image 5</em>
</p>

### <a id="contour-section-3"></a> 3. Compare the results of blob detection and contour detection for the chosen dataset.

<p align="center">
  <img src="results/blob2/blob_vs_contour_detection.png" width="300"/><br>
  <em>Figure 37: Blob- and contour detection applied to images</em>
</p>

Blob detection excels at highlighting regions that stand out from their surroundings. In image 3 of figure 37 (with tram tracks), it better detects the windows on the left building because the pixels differ from the surrounding pixels. In contrast, contour detection is strongest at tracing boundaries and sharp intensity changes, so it better captures the right-hand building’s windows with their well-defined edges. Similar patterns can be seen in the other images.

### <a id="contour-section-4"></a> 4. Discuss the advantages and limitations of each technique.


<p align="center">
  <img src="results/goodblob.png" width="300"/><br>
  <em>Figure 38: Blob and contour detection applied to circular regions</em>
</p>

Blob detection is efficient at identifying roughly circular regions and provides quick localization and size estimates, making it ideal for detecting spots or particles across multiple scales. As shown in Figure 38, blob detection correctly separates the two touching circular objects, whereas contour detection merges them into a single unit. However, the blob detection algorithm lacks detailed shape information and struggles with irregular or complex objects.

<p align="center">
  <img src="results/goodcontour.png" width="300"/><br>
  <em>Figure 39: Blob and contour detection applied to rectangular regions</em>
</p>

Contour detection, on the other hand, excels at outlining precise object boundaries and capturing detailed shape features, which is valuable for morphological analysis. As shown in Figure 38, contour detection correctly identifies a the rectangular object, whereas blob detection splits the region into multiple blobs. The effectiveness of the contour detection technique depends heavily on image quality and edge definition, and it can be computationally more intensive and sensitive to noise.

So blob detection is best for fast, approximate feature localization, while contour detection is preferred when detailed shape and boundary information is required.

### <a id="contour-section-5"></a> 5. Analyze the impact of different parameters (e.g., threshold values, filter sizes) on the detection results.

The performance of both blob and contour detection methods is sensitive to parameters like threshold values and filter sizes.

In blob detection, adjusting the `threshold` controls the sensitivity. Lower thresholds detect more blobs but increase false positives, while higher thresholds reduce noise but may miss subtle features. Similarly, the choice of `max_sigma` and `num_sigma` affects the scale range and granularity of detected blobs. The variations of threshold values in blob detection are shown in figure 25.

<p align="center">
  <img src="results/contour_analysis.png" width="800"/><br>
  <em>Figure 40: Contour detection with different thresholds</em>
</p>

For contour detection, the threshold used in binarization critically impacts which features are segmented, too low of a threshold may merge objects or include noise, while too high may fragment or miss contours. Additionally, morphological operations like removing small objects depend on filter sizes that balance noise reduction against losing small meaningful contours. Contour detection with different thresholds are shown in figure 40.

### <a id="contour-section-6"></a> 6. Provide examples where one technique might be more suitable than the other.

Blob detection is more suitable in applications where the target objects are roughly circular and uniformly bright or dark against the background. For example, detecting cells in microscopy images, stars in astronomical images, or bubbles in fluid simulations. Its strength lies in quick localization and size estimation of round features across different scales.

In contrast, contour detection is ideal when precise object boundaries and shape details are essential, such as in medical image analysis, character recognition, or analyzing irregularly shaped objects like leaves or cracks. It enables detailed morphological analysis, making it preferable when shape complexity and boundary accuracy matter more than speed or rough position.

<div style="page-break-after: always;"></div>

