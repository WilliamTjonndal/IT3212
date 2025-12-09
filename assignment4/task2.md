## Task 2

We picked the social media dataset for clustering. The dataset contains data about online news, such as categories they fit into, sentiment analysis, and their popularity. 

An actual use case for clustering on this dataset is to group online news together with others that are similar, preferring to show users that engage more with news from a particular cluster other news within the same cluster. By showing users news similar to those they usually engage with, it's likely total engagement and user retention would increase.

### Preprocessing

To start off with preprocessing the dataset, we looked at all the columns to get an understanding of what the data represents. The dataset contains several columns that are not useful for our selected use case. The first we removed was `url`, as it's useless for clustering. This is because it's unique and categorical, making it impossible to create clusters from.

We also decided to remove all time-related columns, being `weekday_is_monday`, ..., `weekday_is_sunday`, `is_weekend`, and `timedelta`. The reason for deleting these columns even though they might be useful for clustering is that using clustering with them don't make much sense given our theoretical use case. When recommending similar online news to what you engage with, it wouldn't make sense to take into account which day the news were posted, as recommendations on online platforms should almost always prefer recent news. It doesn't matter what day something was posted if you're recommending something posted within ~2 days anyway.

Using this logic, it might make sense to include `timedelta`, which refers to the days between the article publication and dataset aquisition. The reason we didn't include this column is because this value has a large range, up to several years. This doesn't fit well with our goal, as a recommendation algorithm should almost never recommend articles as old as that. If we were to create a recommendation algorithm based on our clustering results, we would put an external limitation that would heavily prefer recent news instead of including age in our clustering. 

`timedelta` does have an effect on `shares`, as older articles have more time to accumulate shares, but we still chose to includes `shares` when clustering. This is because all data points have a `timedelta` of at least 8 days, so we think all articles have had some time to get a number of shares that would be highly correlated with the number of shares they would have after a few days (This assumes articles gain the most traction/shares when they are recently released, meaning even if older articles have much more time to gain shares, most shares are gained within the first few days of release).

#### Scaling
After removing the columns we don't want to include in clustering, it's time to scale the data. It's important to scale our data so the features with a larger range of values won't be preferred over those with smaller ranges based only on their larger range. Looking at the distribution of the data in figure #, we can see columns referring to shares, like `shares` and `kw_avg_max` (the average of the max amount of shares for articles assigned each keyword) have a much larger range than the rest, meaning they would be likely to overpower the other features. Scaling the data will make all our features have the same scale so each feature's importance will be decided fairly.

<p align="center">
<img src="task2/img/scaling_dist_before.png" width="600"/><br>
<em>Figure #: Distribution of data before scaling</em>
</p>

We chose to use min-max scaling, mostly because it's results are easier to understand the results for columns like `shares` and `num_imgs`, and it preserves the distribution of our data. The results being easier to understand is not really the case for columns representing sentiment analysis, like `global_sentiment_polarity` and `title_subjectivity`, as we don't have an intuitive understanding of what a specific value means, other than in relation to other values. We still chose to use min-max scaling here to keep the same scaling method for all our features, and again to preserve the distribution of all our features. Looking at the distribution of the scaled data in figure #, we can see the distributions now look much more even which should give better results for the clustering. 

<p align="center">
<img src="task2/img/scaling_dist_after.png" width="600"/><br>
<em>Figure #: Distribution of data after scaling</em>
</p>

#### Ourlier detection
When removing outliers from our data, we chose between z-score and IQR. We decided to use IQR as we can see in figure # that the distribution of almost every column is normally distributed, but skewed. After some testing, we noticed removing outliers using IQR on all non-categorical columns would remove way more rows than expected. To keep a larger portion of the dataset, we had to select which features to do outlier detection and removal using.

We decided not to use outlier detection on columns referencing shares or sentiment analysis, as shares are more extremely skewed than other features in the dataset. Features referencing shares include `shares` and `kw_avg_max`. We also decided not to remove outliers using features based on sentiment analysis, as it would be very hard for us to tell if a very high or low value is actually outside the range of what's likely a real data point. Detecting outliers based on a column we don't know a real range of would not be a good idea, as the goal of handling outliers is removing or changing values not generated by the same method as the others.

Performing outlier detection on the remaining columns (not categorical, referencing shares, or based on sentiment analysis), such as `n_tokens_title` and `num_videos`, we were able to find `2360` outliers. Looking at some of these outliers, they contain things things like there being `91` videos or 116 images in an article. This is about 6% of our dataset, which has a total of `39644` rows. We decided to remove the data points containing the outliers, as it's not a very large portion of our dataset.

In figure #, you can see the distribution of our features after removing outliers, while figure # shows the distribution after re-scaling our dataset between 0 and 1. We did outlier detection after scaling to better visualize our results, but it has the same effect as detecting outliers before scaling when using min-max scaling, as the distribution within each feature stays the same.

<p align="center">
<img src="task2/img/iqr_dist.png" width="600"/><br>
<em>Figure #: Distribution of data after removing outliers</em>
</p>

<p align="center">
<img src="task2/img/iqr_dist_rescaled.png" width="600"/><br>
<em>Figure #: Distribution of data after removing outliers and re-scaling</em>
</p>

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

### Clustering methods

There are several different types of clustering algorithms to choose from, each with different properties. There are several types of clustering algorithms to choose from, but we mainly looked at centroid-based, density-based and distribution-based clustering algorithms.\
The simplest are centroid-based algorithms that create cluster centers and assign data points to clusters based on their distance from the cluster centers. Examples of this are K-means and fuzzy C-means.\
Density-based algorithms define clusters based on the density of data points. This fits well for finding clusters of different shapes, but won't work well for clusters with varying densities. DBSCAN is an example of a density-based clustering algorithm.\
Distribution-based algorithms assume clusters are generated by probability distributions, which work well for clusters of different (but not too complicated) shapes and densities, but it does assume data is generated in a specific distribution. An example of distribution-based clustering algorithms is Gaussian Mixture Models.

#### K-means
K-means was chosen mainly because it's the easiest clustering algorithm to understand. A problem with K-means and other centroid based algorithms is that it works best for clusters that are approximately spherical and similar in size, which is not the case for this dataset. This could have a negative impact on the performance of the clustering if the data points don't seperate well with centroid based clustering algorithms.

#### Fuzzy C-means
Fuzzy C-means works almost like an improved version of K-means for this dataset. The biggest difference between them is that fuzzy C-means assigns each data point a membership value for every cluster, meaning it would be better for our use case. Even though articles are seperated into clusters, we don't want to make it impossible for articles that fit better in another cluster to be recommended to users, just less likely. Using fuzzy C-means, we can assign how likely an article is to be recommended to users based on its membership value to each cluster. Aside from that, it works very similarly to K-means as both are centroid-based clustering algorithms.

#### Gaussian Mixture Models
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


### Compare and explain

#### K-means

<p align="center">
<img src="task2/img/clusters_kmeans.png" width="500"/><br>
<em>Figure #: Visualization of K-means clustering</em>
</p>

<p align="center">
<img src="task2/img/features_kmeans.png" width="600"/><br>
<em>Figure #: Top features contributing to K-means clustering</em>
</p>

#### Fuzzy C-means

<p align="center">
<img src="task2/img/clusters_fcm.png" width="500"/><br>
<em>Figure #: Visualization of fuzzy C-means clustering</em>
</p>

<p align="center">
<img src="task2/img/features_fcm.png" width="600"/><br>
<em>Figure #: Top features contributing to fuzzy C-means clustering</em>
</p>

#### Gaussian Mixture Models

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
