## <a id="task-2"></a> Task 2

### <a id="task-2-a"></a> Pick any dataset from the list, implement the preprocessing and justify the preprocessing steps, extract features and justify the methods used, select features and justify the methods used.

We picked the social media dataset for clustering. The dataset contains data about online news, such as categories they fit into, sentiment analysis, and their popularity. 

An actual use case for clustering on this dataset is to group online news together with others that are similar, preferring to show users that engage more with news from a particular cluster other news within the same cluster. By showing users news similar to those they usually engage with, it's likely total engagement and user retention would increase.

<div style="text-align: center">
  <table style="margin-left:auto; margin-right:auto">
    <tr>
      <td>url</td><td>timedelta</td><td>n_tokens_title</td><td>n_tokens_content</td>
    </tr>
    <tr>
      <td>n_unique_tokens</td><td>n_non_stop_words</td><td>n_non_stop_unique_tokens</td><td>num_hrefs</td>
    </tr>
    <tr>
      <td>num_self_hrefs</td><td>num_imgs</td><td>num_videos</td><td>average_token_length</td>
    </tr>
    <tr>
      <td>num_keywords</td><td>data_channel_is_lifestyle</td><td>data_channel_is_entertainment</td><td>data_channel_is_bus</td>
    </tr>
    <tr>
      <td>data_channel_is_socmed</td><td>data_channel_is_tech</td><td>data_channel_is_world</td><td>kw_min_min</td>
    </tr>
    <tr>
      <td>kw_max_min</td><td>kw_avg_min</td><td>kw_min_max</td><td>kw_max_max</td>
    </tr>
    <tr>
      <td>kw_avg_max</td><td>kw_min_avg</td><td>kw_max_avg</td><td>kw_avg_avg</td>
    </tr>
    <tr>
      <td>self_reference_min_shares</td><td>self_reference_max_shares</td><td>self_reference_avg_sharess</td><td>weekday_is_monday</td>
    </tr>
    <tr>
      <td>weekday_is_tuesday</td><td>weekday_is_wednesday</td><td>weekday_is_thursday</td><td>weekday_is_friday</td>
    </tr>
    <tr>
      <td>weekday_is_saturday</td><td>weekday_is_sunday</td><td>is_weekend</td><td>LDA_00</td>
    </tr>
    <tr>
      <td>LDA_01</td><td>LDA_02</td><td>LDA_03</td><td>LDA_04</td>
    </tr>
    <tr>
      <td>global_subjectivity</td><td>global_sentiment_polarity</td><td>global_rate_positive_words</td><td>global_rate_negative_words</td>
    </tr>
    <tr>
      <td>rate_positive_words</td><td>rate_negative_words</td><td>avg_positive_polarity</td><td>min_positive_polarity</td>
    </tr>
    <tr>
      <td>max_positive_polarity</td><td>avg_negative_polarity</td><td>min_negative_polarity</td><td>max_negative_polarity</td>
    </tr>
    <tr>
      <td>title_subjectivity</td><td>title_sentiment_polarity</td><td>abs_title_subjectivity</td><td>abs_title_sentiment_polarity</td>
    </tr>
    <tr>
      <td>shares</td><td></td><td></td><td></td>
    </tr>
  </table>
  <br>
  <em>Figure #: All columns in the dataset</em>
</div>


To start off with preprocessing the dataset, we looked at all the columns to get an understanding of what the data represents. The columns are showin in figure #. Features with names starting with `kw_` represent the amount of shares gained by articles assigned each keyword, looking at the min, average, and max shares for the best, average, and worst keywords associated with the article. Features starting with `LDA_` represent closeness to a given LDA topic (abstract topics/themes decided by another machine learning algorithm). Many of the features, such as `global_sentiment_polarity` and `title_subjectivity` are based on sentiment analysis.

The dataset contains several columns that are not useful for our selected use case. The first we removed was `url`, as it's useless for clustering. This is because it's unique and categorical, making it impossible to create clusters from.

We also decided to remove all time-related columns, being `weekday_is_monday`, ..., `weekday_is_sunday`, `is_weekend`, and `timedelta`. The reason for deleting these columns even though they might be useful for clustering is that using clustering with them don't make much sense given our theoretical use case. When recommending similar online news to what you engage with, it wouldn't make sense to take into account which day the news were posted, as recommendations on online platforms should almost always prefer recent news. It doesn't matter what day something was posted if you're recommending something posted within ~2 days anyway.

Using this logic, it might make sense to include `timedelta`, which refers to the days between the article publication and dataset aquisition. The reason we didn't include this column is because this value can range from a few days to several years, which is a much larger range than what would be expected when recommending online news. If we were to create a recommendation algorithm based on our clustering results, we would put an external limitation that would heavily prefer recent articles instead of including `timedelta` when clustering. 

`timedelta` does have an effect on `shares`, as older articles have more time to accumulate shares, but we still chose to includes `shares` when clustering. This is because all data points have a `timedelta` of at least 8 days, so we think all articles have had some time to get a number of shares that would be highly correlated with the number of shares they would have after a few days (This assumes articles gain the most traction/shares when they are recently released, meaning even if older articles have much more time to gain shares, most shares are gained within the first few days of release).

After removing some columns, we are left with the features showin in figure #, along with their some data on their distributions. We can also see from this figure that there are no missing values, as each column contains the same count as the total number of rows.

<div style="text-align: center">
  <table style="margin-left:auto; margin-right:auto">
    <tr>
      <th>Feature</th><th>Count</th><th>Mean</th><th>Std</th><th>Min</th><th>25%</th><th>50%</th><th>75%</th><th>Max</th>
    </tr>
    <tr><td>n_tokens_title</td><td>39644</td><td>10.398749</td><td>2.114037</td><td>2</td><td>9</td><td>10</td><td>12</td><td>23</td></tr>
    <tr><td>n_tokens_content</td><td>39644</td><td>546.514731</td><td>471.107508</td><td>0</td><td>246</td><td>409</td><td>716</td><td>8474</td></tr>
    <tr><td>n_unique_tokens</td><td>39644</td><td>0.548216</td><td>3.520708</td><td>0</td><td>0.470870</td><td>0.539226</td><td>0.608696</td><td>701</td></tr>
    <tr><td>n_non_stop_words</td><td>39644</td><td>0.996469</td><td>5.231231</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1042</td></tr>
    <tr><td>n_non_stop_unique_tokens</td><td>39644</td><td>0.689175</td><td>3.264816</td><td>0</td><td>0.625739</td><td>0.690476</td><td>0.754630</td><td>650</td></tr>
    <tr><td>num_hrefs</td><td>39644</td><td>10.883690</td><td>11.332017</td><td>0</td><td>4</td><td>8</td><td>14</td><td>304</td></tr>
    <tr><td>num_self_hrefs</td><td>39644</td><td>3.293638</td><td>3.855141</td><td>0</td><td>1</td><td>3</td><td>4</td><td>116</td></tr>
    <tr><td>num_imgs</td><td>39644</td><td>4.544143</td><td>8.309434</td><td>0</td><td>1</td><td>1</td><td>4</td><td>128</td></tr>
    <tr><td>num_videos</td><td>39644</td><td>1.249874</td><td>4.107855</td><td>0</td><td>0</td><td>0</td><td>1</td><td>91</td></tr>
    <tr><td>average_token_length</td><td>39644</td><td>4.548239</td><td>0.844406</td><td>0</td><td>4.478404</td><td>4.664082</td><td>4.854839</td><td>8.041534</td></tr>
    <tr><td>num_keywords</td><td>39644</td><td>7.223767</td><td>1.909130</td><td>1</td><td>6</td><td>7</td><td>9</td><td>10</td></tr>
    <tr><td>data_channel_is_lifestyle</td><td>39644</td><td>0.052946</td><td>0.223929</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td></tr>
    <tr><td>data_channel_is_entertainment</td><td>39644</td><td>0.178009</td><td>0.382525</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td></tr>
    <tr><td>data_channel_is_bus</td><td>39644</td><td>0.157855</td><td>0.364610</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td></tr>
    <tr><td>data_channel_is_socmed</td><td>39644</td><td>0.058597</td><td>0.234871</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td></tr>
    <tr><td>data_channel_is_tech</td><td>39644</td><td>0.185299</td><td>0.388545</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td></tr>
    <tr><td>data_channel_is_world</td><td>39644</td><td>0.212567</td><td>0.409129</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td></tr>
    <tr><td>kw_min_min</td><td>39644</td><td>26.106801</td><td>69.633215</td><td>-1</td><td>-1</td><td>-1</td><td>4</td><td>377</td></tr>
    <tr><td>kw_max_min</td><td>39644</td><td>1153.951682</td><td>3857.990877</td><td>0</td><td>445</td><td>660</td><td>1000</td><td>298400</td></tr>
    <tr><td>kw_avg_min</td><td>39644</td><td>312.366967</td><td>620.783887</td><td>-1</td><td>141.750000</td><td>235.500000</td><td>357</td><td>42827.857143</td></tr>
    <tr><td>kw_min_max</td><td>39644</td><td>13612.354102</td><td>57986.029357</td><td>0</td><td>0</td><td>1400</td><td>7900</td><td>843300</td></tr>
    <tr><td>kw_max_max</td><td>39644</td><td>752324.066694</td><td>214502.129573</td><td>0</td><td>843300</td><td>843300</td><td>843300</td><td>843300</td></tr>
    <tr><td>kw_avg_max</td><td>39644</td><td>259281.938083</td><td>135102.247285</td><td>0</td><td>172846.875</td><td>244572.222</td><td>330980</td><td>843300</td></tr>
    <tr><td>kw_min_avg</td><td>39644</td><td>1117.146610</td><td>1137.456951</td><td>-1</td><td>0</td><td>1023.635611</td><td>2056.781032</td><td>3613.039819</td></tr>
    <tr><td>kw_max_avg</td><td>39644</td><td>5657.211151</td><td>6098.871957</td><td>0</td><td>3562.101631</td><td>4355.688836</td><td>6019.953968</td><td>298400</td></tr>
    <tr><td>kw_avg_avg</td><td>39644</td><td>3135.858639</td><td>1318.150397</td><td>0</td><td>2382.448566</td><td>2870.074878</td><td>3600.229564</td><td>43567.659946</td></tr>
    <tr><td>self_reference_min_shares</td><td>39644</td><td>3998.755396</td><td>19738.670516</td><td>0</td><td>639</td><td>1200</td><td>2600</td><td>843300</td></tr>
    <tr><td>self_reference_max_shares</td><td>39644</td><td>10329.212662</td><td>41027.576613</td><td>0</td><td>1100</td><td>2800</td><td>8000</td><td>843300</td></tr>
    <tr><td>self_reference_avg_sharess</td><td>39644</td><td>6401.697580</td><td>24211.332231</td><td>0</td><td>981.1875</td><td>2200</td><td>5200</td><td>843300</td></tr>
    <tr><td>LDA_00</td><td>39644</td><td>0.184599</td><td>0.262975</td><td>0</td><td>0.025051</td><td>0.033387</td><td>0.240958</td><td>0.926994</td></tr>
    <tr><td>LDA_01</td><td>39644</td><td>0.141256</td><td>0.219707</td><td>0</td><td>0.025012</td><td>0.033345</td><td>0.150831</td><td>0.925947</td></tr>
    <tr><td>LDA_02</td><td>39644</td><td>0.216321</td><td>0.282145</td><td>0</td><td>0.028571</td><td>0.040004</td><td>0.334218</td><td>0.919999</td></tr>
    <tr><td>LDA_03</td><td>39644</td><td>0.223770</td><td>0.295191</td><td>0</td><td>0.028571</td><td>0.040001</td><td>0.375763</td><td>0.926534</td></tr>
    <tr><td>LDA_04</td><td>39644</td><td>0.234029</td><td>0.289183</td><td>0</td><td>0.028574</td><td>0.040727</td><td>0.399986</td><td>0.927191</td></tr>
    <tr><td>global_subjectivity</td><td>39644</td><td>0.443370</td><td>0.116685</td><td>0</td><td>0.396167</td><td>0.453457</td><td>0.508333</td><td>1</td></tr>
    <tr><td>global_sentiment_polarity</td><td>39644</td><td>0.119309</td><td>0.096931</td><td>-0.39375</td><td>0.057757</td><td>0.119117</td><td>0.177832</td><td>0.727841</td></tr>
    <tr><td>global_rate_positive_words</td><td>39644</td><td>0.039625</td><td>0.017429</td><td>0</td><td>0.028384</td><td>0.039023</td><td>0.050279</td><td>0.155488</td></tr>
    <tr><td>global_rate_negative_words</td><td>39644</td><td>0.016612</td><td>0.010828</td><td>0</td><td>0.009615</td><td>0.015337</td><td>0.021739</td><td>0.184932</td></tr>
    <tr><td>rate_positive_words</td><td>39644</td><td>0.682150</td><td>0.190206</td><td>0</td><td>0.600000</td><td>0.710526</td><td>0.800000</td><td>1</td></tr>
    <tr><td>rate_negative_words</td><td>39644</td><td>0.287934</td><td>0.156156</td><td>0</td><td>0.185185</td><td>0.280000</td><td>0.384615</td><td>1</td></tr>
    <tr><td>avg_positive_polarity</td><td>39644</td><td>0.353825</td><td>0.104542</td><td>0</td><td>0.306244</td><td>0.358755</td><td>0.411428</td><td>1</td></tr>
    <tr><td>min_positive_polarity</td><td>39644</td><td>0.095446</td><td>0.071315</td><td>0</td><td>0.050</td><td>0.100</td><td>0.100</td><td>1</td></tr>
    <tr><td>max_positive_polarity</td><td>39644</td><td>0.756728</td><td>0.247786</td><td>0</td><td>0.600</td><td>0.800</td><td>1.000</td><td>1</td></tr>
    <tr><td>avg_negative_polarity</td><td>39644</td><td>-0.259524</td><td>0.127726</td><td>-1</td><td>-0.328383</td><td>-0.253333</td><td>-0.186905</td><td>0</td></tr>
    <tr><td>min_negative_polarity</td><td>39644</td><td>-0.521944</td><td>0.290290</td><td>-1</td><td>-0.700000</td><td>-0.500000</td><td>-0.300000</td><td>0</td></tr>
    <tr><td>max_negative_polarity</td><td>39644</td><td>-0.107500</td><td>0.095373</td><td>-1</td><td>-0.125000</td><td>-0.100000</td><td>-0.050000</td><td>0</td></tr>
    <tr><td>title_subjectivity</td><td>39644</td><td>0.282353</td><td>0.324247</td><td>0</td><td>0.000000</td><td>0.150000</td><td>0.500000</td><td>1</td></tr>
    <tr><td>title_sentiment_polarity</td><td>39644</td><td>0.071425</td><td>0.265450</td><td>-1</td><td>0.000000</td><td>0.000000</td><td>0.150000</td><td>1</td></tr>
    <tr><td>abs_title_subjectivity</td><td>39644</td><td>0.341843</td><td>0.188791</td><td>0</td><td>0.166667</td><td>0.500000</td><td>0.500000</td><td>0.5</td></tr>
    <tr><td>abs_title_sentiment_polarity</td><td>39644</td><td>0.156064</td><td>0.226294</td><td>0</td><td>0.000000</td><td>0.000000</td><td>0.250000</td><td>1</td></tr>
    <tr><td>shares</td><td>39644</td><td>3395.380184</td><td>11626.950749</td><td>1</td><td>946</td><td>1400</td><td>2800</td><td>843300</td></tr>
  </table>
  <br>
  <em>Figure #: Distribution statistics for all used features</em>
</div>

#### Scaling
After removing the columns we don't want to include in clustering, it's time to scale the data. It's important to scale our data so the features with a larger range of values won't be preferred over those with smaller ranges based only on their larger range. Looking at the distribution of the data in figure #, we can see columns referring to shares, like `shares` and `kw_avg_max` have a much larger range than the rest, meaning they would be likely to overpower the other features. Scaling the data will make all our features have the same scale so each feature's importance will be decided fairly.

We chose to use min-max scaling, mostly because it's results are easier to understand for columns like `shares` and `num_imgs`, and it preserves the distribution of our data. The results being easier to understand is not really the case for columns representing sentiment analysis, like `global_sentiment_polarity` and `title_subjectivity`, as we don't have an intuitive understanding of what a specific value means, other than in relation to other values. We still chose to use min-max scaling here to keep the same scaling method for all our features, and again to preserve the distribution of all our features. Looking at the distribution of the scaled data in figure #, we can see the distributions now look much more even than before scaling in figure #, which should give better results when clustering.

<table>
  <tr>
    <td align="center">
      <img src="task2/img/scaling_dist_before.png" width="500"/><br>
      <em>Figure #: Distribution of data before scaling</em>
    </td>
    <td align="center">
      <img src="task2/img/scaling_dist_after.png" width="500"/><br>
      <em>Figure #: Distribution of data after scaling</em>
    </td>
  </tr>
</table>

#### Outlier detection
The first outliers we checked for were values outside the possible range for each feature. We found negative values in `kw_min_min`, `kw_avg_min`, and `kw_min_avg`, as seen in figure #. This is not possible as each of these columns refer to an amount of shares articles with a given keyword has received. Since shares can't be negative, we decided to cap the lower value of these columns to 0.

When removing outliers based on distribution from our data, we chose between using z-score and IQR. We decided to use IQR as we can see in figure # that the distribution of almost every column is not normally distributed, but skewed. After some testing, we noticed removing outliers using IQR on all non-categorical columns would remove way more rows than expected. To keep a larger portion of the dataset, we had to select which features to use for outlier detection and removal.

We decided not to use outlier detection on columns referencing shares or sentiment analysis, as shares are more extremely skewed than other features in the dataset, which intuitively makes sense, since some articles become way more popular than others. Features referencing shares include `shares` and `kw_avg_max`. We also decided not to remove outliers using features based on sentiment analysis, as it would be very hard for us to tell if a very high or low value is actually outside the range of what's likely a real data point. Detecting outliers based on a column we don't know a real range of would not be a good idea, as the goal of handling outliers is removing or changing values not generated by the same method as the others.

Performing outlier detection on the remaining columns (not categorical, referencing shares, or based on sentiment analysis), such as `n_tokens_title` and `num_videos`, we were able to find `5465` outliers. Looking at some of these outliers, they contain things like there being `91` videos or `116` images in an article. This is about 13.7% of our dataset, which has a total of `39644` rows. We decided to remove the data points containing the outliers, because even though it's a sizable portion of our dataset, our chosen clustering algorithms don't include outlier detection themselves, and using 86.3% of the data is still plenty for clustering.

In figure #, you can see the distribution of our features after removing outliers, while figure # shows the distribution after re-scaling our dataset between 0 and 1. These figures show the scaled versions of our features for visualization only, and outliers were removed from the original unscaled dataset. The numerical distributions of the features used in outlier detection are also shown before in figure #, and after in figure #.

<table>
  <tr>
    <td align="center">
      <img src="task2/img/iqr_dist.png" width="500"/><br>
      <em>Figure #: Distribution of data after removing outliers</em>
    </td>
    <td align="center">
      <img src="task2/img/iqr_dist_rescaled.png" width="500"/><br>
      <em>Figure #: Distribution of data after removing outliers and re-scaling</em>
    </td>
  </tr>
</table>

<p align="center">
<img src="task2/img/iqr_dist_num_before.png" width="800"/><br>
<em>Figure #: Numerical distribution of data before removing outliers</em>
</p>

<p align="center">
<img src="task2/img/iqr_dist_num_after.png" width="800"/><br>
<em>Figure #: Numerical distribution of data after removing outliers</em>
</p>

#### Dimensionality reduction
We decided to reduce the dimensions of our dataset, as we think it will improve the performance of our clustering methods, as well as giving us a better visualization of our data. Our biggest reason for thinking reducing dimensions will give better performance when clustering is that distance between points becomes less useful the more dimensions are used. Due to how euclidian distance is calculated, the distance between every point converges as dimensions increase, meaning the more dimensions there are in the dataset, the less variation there is in the distance between each point. If every point is almost the same distance from eachother, it becomes very hard to seperate them into meaningful clusters. This problem is a bit exaggerated as it has a much more noticable impact with 100+ dimensions, but it's still better to reduce the dimensions to minimize this effect.

For dimensionality reduction, we thought about using PCA and t-SNE. PCA focuses on keeping as much of the variance in the data as possible, while t-SNE tries to keep higher-dimensional neighbors close even in lower dimensions. While t-SNE sound like a good fit for our dataset, it has a problem which makes it unsuitable for our selected use case. t-SNE finds similarities between all points in the dataset, which works well for preserving neighborhoods, but makes no mapping function that can be used for future data points. This means that if we want to add a new data point (such as a new article being created), we would have to redo our dimensionality reduction on all our data. Due to this, we chose to use PCA for dimensionality reduction, keeping enough principal components to preserve 95% of the variance of the data. This leaves us with 22 out of the 51 original features remaining, as seen in figure #.

<p align="center">
<img src="task2/img/pca_95.png" width="500"/><br>
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

Like fuzzy C-means, GMM assigns a probability to each article for belonging to every cluster. This way, this clustering algorithm doens't force articles that could belong to multiple clusters into just one.

A problem shared between all our selected clustering algorithms is that you have to specify the amount of clusters created for each of them. While it would be nice to use a clustering method that doesn't depend on the amount of clusters, we think we were able to pick a good value for the by seeing how different values performed on certain metrics for each clustering algorithm.

### <a id="task-2-c"></a> Compare and Explain the results

#### <a id="compare-k-means"></a> K-means

To decide the amount of clusters to use for K-means, we found 2 common clusterings metrics to see their performance for each value of k in what we though was a reasonable range based on the visualization, 3-6. The metrics we chose to use were silhouette score and Davies-Bouldin index. Silhouette score measures how similar points within a cluster are to eachother, ranging from 1 to -1. Scores near 1 mean data points are very similar within a cluster compared to data points in other clusters, while scores near -1 are likely misclassified. David-Bouldin index measures how well defined clusters are, comparing their compactness and separation. A value close to 0 means clusters are well separated and compact, while high scores imply overlapping or scattered clusters.

<table align="center">
  <tr>
    <td align="center">
      <img src="task2/img/kmeans_3.png" width="300"/><br>
      <em>Figure #: K-means clustering with k=3</em>
    </td>
    <td align="center">
      <img src="task2/img/kmeans_4.png" width="300"/><br>
      <em>Figure #: K-means clustering with k=4</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="task2/img/kmeans_5.png" width="300"/><br>
      <em>Figure #: K-means clustering with k=5</em>
    </td>
    <td align="center">
      <img src="task2/img/kmeans_6.png" width="300"/><br>
      <em>Figure #: K-means clustering with k=6</em>
    </td>
  </tr>
</table>

<p align="center">
<img src="task2/img/kmeans_clusters_score.png" width="650"/><br>
<em>Figure #: K-means clustering scores with different k</em>
</p>

We can see from figure # that the best performing amount of clusters is 4, both maximizing its silhouette score and minimizing its Davies-Bouldin index. Using this value, we find the final clustered dataset using K-means in figure #.

<p align="center">
<img src="task2/img/kmeans_4.png" width="650"/><br>
<em>Figure #: Visualization of K-means clustering</em>
</p>

We can also tell which features from our dataset have the largest effect on placing data points into clusters, the most influential features for clustering can be seen in figure #.

<p align="center">
<img src="task2/img/features_kmeans.png" width="650"/><br>
<em>Figure #: Top features contributing to K-means clustering</em>
</p>

#### <a id="compare-fuzzy-c-means"></a> Fuzzy C-means

<table align="center">
  <tr>
    <td align="center">
      <img src="task2/img/fcm_3.png" width="300"/><br>
      <em>Figure #: Fuzzy C-means clustering with 3 clusters</em>
    </td>
    <td align="center">
      <img src="task2/img/fcm_4.png" width="300"/><br>
      <em>Figure #: Fuzzy C-means clustering with 4 clusters</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="task2/img/fcm_5.png" width="300"/><br>
      <em>Figure #: Fuzzy C-means clustering with 5 clusters</em>
    </td>
    <td align="center">
      <img src="task2/img/fcm_6.png" width="300"/><br>
      <em>Figure #: Fuzzy C-means clustering with 6 clusters</em>
    </td>
  </tr>
</table>

<p align="center">
<img src="task2/img/fcm_clusters_score.png" width="650"/><br>
<em>Figure #: Fuzzy C-means clustering scores with different amounts of clusters</em>
</p>

We also found 4 clusters to be the optimal amount for fuzzy C-means, as seen in figure #, which is to be expected since it works very similarly to K-means, both being centroid-based clustering algorithms. The visualization of the optimal fuzzy C-means clustering is seein in figure #.

<p align="center">
<img src="task2/img/fcm_4.png" width="650"/><br>
<em>Figure #: Visualization of fuzzy C-means clustering</em>
</p>

<p align="center">
<img src="task2/img/features_fcm.png" width="650"/><br>
<em>Figure #: Top features contributing to fuzzy C-means clustering</em>
</p>

#### <a id="compare-gaussian-mixture-models"></a> Gaussian mixture models

<table align="center">
  <tr>
    <td align="center">
      <img src="task2/img/gmm_3.png" width="300"/><br>
      <em>Figure #: GMM clustering with 3 clusters</em>
    </td>
    <td align="center">
      <img src="task2/img/gmm_4.png" width="300"/><br>
      <em>Figure #: GMM clustering with 4 clusters</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="task2/img/gmm_5.png" width="300"/><br>
      <em>Figure #: GMM clustering with 5 clusters</em>
    </td>
    <td align="center">
      <img src="task2/img/gmm_6.png" width="300"/><br>
      <em>Figure #: GMM clustering with 6 clusters</em>
    </td>
  </tr>
</table>

<p align="center">
<img src="task2/img/gmm_clusters_score.png" width="650"/><br>
<em>Figure #: GMM clustering scores with different amounts of clusters</em>
</p>

We also found 4 as the optimal number of clusters for GMM, seen in figure #. After finding this, we created the visualization of GMM using 4 clusters shown in figure #.

<p align="center">
<img src="task2/img/gmm_4.png" width="650"/><br>
<em>Figure #: Visualization of GMM clustering</em>
</p>

<p align="center">
<img src="task2/img/features_gmm.png" width="650"/><br>
<em>Figure #: Top features contributing to GMM clustering</em>
</p>

All our chosen clustering algorithms had very similar visualizations, but the differences are more clear when looking at the clustering metrics. Fuzzy C-means had the best performance, reaching a silhouette score of 0.647 and a Davies-Bouldin index of 0.500. This is a slight performance increase from K-means, and a larger jump from GMM. Part of the reason for this result is likely that both silhouette score and Davies-Bouldin index favor centroid-based clustering, as they both rely on euclidian distance, favoring compact spherical clusters. Both metrics are still valid metrics for GMM and some of the easier to understand among clustering metrics. They also have the advantage of being possible to measure for all our chosen clustering algorihtms, which isn't the case for all metrics.

It's also important to remember that fuzzy C-means and GMM has the additional advantage of giving points membership values instead of placing them only on one cluster, which we think make both of these algorightms a better fit for our use case than K-means, even if looking at the clustering metrics, it has better performance than GMM.

We can also see the top features contributing to the selection of clusters by each algorithm. The results here are very similar to eachother, mainly focusing on categories like `data_channel_is_world` and `LDA_00`. The weights for each feature is different between the algorithms, but there are no major changes.

A lot of the similarity here likely come from how the original features were translated into princial components when doing PCA, and you might see a more varied set of contributiong features if using the base data. Even if the results are very similar and mostly using the same few features to base a data point's cluster on, this actually fits very well for our use case. Splitting articles mainly by categories and topics makes the most sense to create a useful recommendation algorithm. Had our results been that clusters were mainly decided by `shares` and similar features, this would probably not show users articles they are as interested in as with our current main clustering factors.

From looking at our performance metrics, we chose to select fuzzy C-means as our preferred clustering algorithm for this dataset. After selecting our preferred clustering algorithml, we took a look at some of the data points most confidentally placed in each cluster to see how similar they were. The top 10 contributing features for 5 points in each cluster can be seen in figure #. It's clear from the points that the clustering has worked well, placing data points in clusters along with other points that are very similar to them in multiple columns.

<p align="center">
<img src="task2/img/fcm_examples.png" width="800"/><br>
<em>Figure #: The 5 most confidently placed data points in each cluster</em>
</p>
