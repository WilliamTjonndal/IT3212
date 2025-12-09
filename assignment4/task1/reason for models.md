sampling - sounds cool, technical reasons
`XGBoost` - familiar model
`RandomForest` - familiar model, easy to use and understand

**RandomForest**
`RandomForest` is a reliable ensemble method that works very well with hand-crafted features like HOG and LBP because it naturally handles high-dimensional inputs and nonlinear relationships without extra preprocessing. Its interpretability and resistance to overfitting make it a strong baseline for comparing different feature extraction settings. For our six scene classes, `RandomForest` performs well on categories with strong structural cues—such as buildings and street—because the trees easily separate features tied to edges and geometric patterns. However, it may struggle with visually similar natural classes like forest, mountain, and glacier, since tree splits can miss subtle continuous gradients in texture and appearance.

**XGBoost**
`XGBoost` was chosen because its gradient-boosted tree structure captures complex interactions between HOG, LBP, and combined features better than bagging-based methods. Its regularization and iterative boosting process help it achieve strong generalization even with large, mixed feature sets. For our scene types, `XGBoost` handles fine distinctions—such as mountain vs. glacier or glacier vs. sea—more effectively than `RandomForest` because boosting gradually corrects misclassifications. The downside is that `XGBoost` is more sensitive to hyperparameters and more computationally demanding, especially when scenes contain highly variable textures like forest landscapes.

**Sampling-based Neural Model**
A `sampling-based neural model` offers adaptive feature learning by extracting useful patterns directly from pixel data rather than relying on predefined descriptors. This allows it to learn both local textures and high-level structures efficiently while remaining lighter than full CNNs. For our six image categories, such a model can learn rich features that distinguish both natural scenes (forest, mountain, glacier) and man-made ones (buildings, street) more flexibly than HOG or LBP. Its main limitations are higher data requirements and the risk of missing key visual information if sampled regions fail to capture important scene elements, such as skylines or water boundaries.
