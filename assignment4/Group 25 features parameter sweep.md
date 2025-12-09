# RandomForest

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

# Stacking

# SVM