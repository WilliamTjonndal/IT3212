**RandomForest**
`RandomForest` is a reliable ensemble method that works very well with hand-crafted features like HOG and LBP because it naturally handles high-dimensional inputs and nonlinear relationships without extra preprocessing. Its interpretability and resistance to overfitting make it a strong baseline for comparing different feature extraction settings. For our six scene classes, `RandomForest` performs well on categories with strong structural cues—such as buildings and street—because the trees easily separate features tied to edges and geometric patterns. However, it may struggle with visually similar natural classes like forest, mountain, and glacier, since tree splits can miss subtle continuous gradients in texture and appearance.

**XGBoost**
`XGBoost` was chosen because its gradient-boosted tree structure captures complex interactions between HOG, LBP, and combined features better than bagging-based methods. Its regularization and iterative boosting process help it achieve strong generalization even with large, mixed feature sets. For our scene types, `XGBoost` handles fine distinctions—such as mountain vs. glacier or glacier vs. sea—more effectively than `RandomForest` because boosting gradually corrects misclassifications. The downside is that `XGBoost` is more sensitive to hyperparameters and more computationally demanding, especially when scenes contain highly variable textures like forest landscapes.

**Stacking**

We chose `stacking` because it lets us combine models that capture different aspects of our scene-classification task—*buildings, forest, glacier, mountain, sea,* and *street*—resulting in a more balanced and robust classifier. `Random Forest` handles nonlinear patterns and noisy HOG/LBP features well, while `SVM` provides strong margin-based separation in high-dimensional space. Using `logistic regression` as the meta-learner keeps the final decision simple, well-regularized, and less prone to overfitting. Beyond sounding like a cool, advanced technique, stacking gives us a technically strong way to merge complementary strengths into a single, more reliable model.

**SVM**

We selected ``SVM (Support Vector Machines)`` because it's a method that we're familiar with and it performs reliably on high-dimensional features like HOG and LBP. Its ability to create strong separating margins—and, with kernels, capture subtle nonlinear differences—makes it effective for visually overlapping scene categories such as *mountain* vs. *glacier* or *street* vs. *buildings*. This combination of technical suitability and practical familiarity allows us to tune and interpret the model confidently while achieving strong performance on diverse image classes.

