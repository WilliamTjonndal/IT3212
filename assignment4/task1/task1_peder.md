Implementing a CNN with hyperparameter tuning

Convolutional Neural Networks (CNNs) are a class of deep learning models specifically designed to exploit the spatial structure in image data. Instead of treating each pixel as an independent feature (as in traditional machine learning models), CNNs use convolutional filters and pooling operations to learn hierarchical feature representations directly from the raw image. This makes them particularly well suited for image classification, compared to models such as Random Forests, XGBoost, or stacking ensembles which typically rely on hand-crafted and/or pre-computed features.

In our experiments, we implemented a CNN in TensorFlow/Keras and trained it on the preprocessed image data. The model consisted of several convolutional and max-pooling layers followed by fully connected layers and a final softmax output over the six classes. To increase robustness and enlarge the effective training set, we applied data augmentation to the training images (horizontal flipping, affine “skewing”, and central cropping followed by resizing). Importantly, this augmentation step was the only additional preprocessing performed for the CNN; we did not perform separate feature extraction as we did for the basic and advance models.

The baseline CNN (with a fixed architecture and reasonable default hyperparameters) trained on the full augmented dataset in approximately 15 minutes on CPU. To investigate the effect of hyperparameters, we then performed a grid search over multiple CNN configurations. This hyperparameter tuning was substantially more expensive: even when using only 10% of the training data, the grid search took more than two hours to complete. The tuned model also showed clear signs of overfitting: it achieved a validation accuracy of 67.8%, while the final test accuracy dropped to 61%. This suggests that the hyperparameter search found a configuration that fit the validation split too closely, without improving generalization to unseen data.

By contrast, when we trained our “standard” CNN model (without the heavy grid search) on 100% of the available data, we obtained a substantially higher test accuracy of 83%. This underlines two important points: (i) CNNs can leverage larger amounts of raw image data effectively thanks to their ability to learn features end-to-end, and (ii) hyperparameter tuning must be done carefully to avoid overfitting to a particular validation set, especially when the tuning budget is large compared to the size of the dataset.

Comparison and explanation of computation time and classification performance

When comparing the different classification algorithms, both computation time and predictive performance showed clear differences between the CNN and the traditional machine learning models (Random Forest, XGBoost, and the stacking ensemble).

For the basic and advanced models, we did not feed raw images directly. Instead, we first computed feature representations for each image. This feature extraction pipeline took roughly 30 minutes to run. On top of this, training the basic and advanced models on just 20% od the data exceeded the total training time of the CNN on the full dataset, even with data augmentation.

There are several plausible reasons why Random Forest, XGBoost, and the stacking ensemble required more computation time than the CNN:

Extra preprocessing cost:
The feature extraction step for RF and XGBoost is a separate stage that must be applied to every image before training. In contrast, the CNN learns features directly from the raw pixels, and the only additional preprocessing, data augmentation, is done before its given to the model.

Algorithmic differences:
Tree-based methods build many decision trees. Each tree involves repeated splitting of the data based on feature values, which is relatively expensive on CPUs. A CNN, on the other hand, mainly convolutions and matrix multiplications, which libraries like TensorFlow optimize heavily, using paralellization.

Model complexity over multiple models (stacking):
The stacking model combines predictions from several base learners (e.g., Random Forest and SVM) into a meta-model (logistic regression). This effectively multiplies the training cost: each base model must be trained, predictions must be computed, and then the meta-model must be fitted. This multi-stage procedure is naturally more time-consuming than training a single CNN end-to-end. However it performed better than its base learners. Stacking can have the benefit of better accuracy by learning from the predictions of its base learners, thereby making its predictions more reliable. Furthermore, stacking allows the meta model to make use of its base learners strengths, and recognize their mistakes. Lastly, stacking is also highly flexible to different problems. However, as in our case, it is computationally heavy, even with only two base learners. If a problem requires a quick solution and deployment, this strategy would not be adviced.

In terms of classification performance, the CNN clearly outperformed both Random Forest and XGBoost on this image classification task. While the tuned CNN variant overfitted and achieved only 61% test accuracy, our main CNN model trained on 100% of the data reached 83% accuracy, compared to substantially lower accuracies for the tree-based models.

This performance gap can be explained by how the models use information in the data:

Exploiting spatial structure:
CNNs operate directly on 2D image grids with multiple channels and use convolutional filters to capture local patterns such as edges, textures, and shapes. Deeper layers combine these into higher-level concepts. This hierarchical feature learning is very effective for images. We think that maybe the features that the cnn extracts on its own, might be better features compared to the ones we precomputed for our other models.

Information loss in feature extraction:
Furthermore, for Random Forest, XGBoost ans the stacking ensemble, we reduced each image to a small set of hand-crafted features (three feature extractions per image). While this drastically reduces dimensionality and makes the models easier to train, it also loses spatial and textural information present in the original images. If the chosen features are not expressive enough, the models are simply not given sufficient information to match the CNN’s performance.

Model capacity and flexibility:
The CNN has a high capacity to approximate complex decision boundaries directly in pixel space, while the tree-based models are constrained to operate on a small, fixed feature vector. Even powerful ensemble methods like XGBoost will be limited by the quality and richness of those features. In our experiments, this likely led to a situation where the CNN could capture more nuanced visual patterns and therefore generalize better on the test set.

Overall, the results show that, despite the common perception that deep learning models are always slower and more resource-intensive, a reasonably sized CNN can be competitive or even faster than traditional methods in practice—especially when the latter depend on expensive feature extraction pipelines. At the same time, the CNN achieved clearly superior classification performance on this image dataset, which is consistent with its architectural advantages for image-based tasks.