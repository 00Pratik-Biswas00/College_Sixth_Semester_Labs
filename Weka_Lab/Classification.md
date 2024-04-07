
# Week-3: Demonstrate performing classification on data sets.

Questions:

#### [A. Apply J48 classifier to the following list of data sets: iris, diabetes, segment-challenge, breast cancer, and report your output. How do you control the growth of the decision tree while using J48 algorithm?](#section-1)

#### [B. Describe and explain each of the classification metrics reported by Weka.](#section-2)

## A. Apply J48 classifier to the following list of data sets: iris, diabetes, segment-challenge, breast cancer, and report your output. How do you control the growth of the decision tree while using J48 algorithm? ------------------------- <a name="section-1"></a>

First part -

Load the dataset, 2. Go to the Classification section and choose the J48 classifier, check the results, by right-clicking on the J48 option you can see the Visualize Tree option and check the tree.

Second part -

In Weka's implementation of the J48 algorithm (C4.5 decision tree), you can control the growth of the decision tree using several parameters. Here are some common options:

Minimum Number of Instances to Split (MinNumObj): This parameter specifies the minimum number of instances that must be present in a node for it to be considered for splitting further. Increasing this value can prevent the tree from growing too large and overfitting the training data.

Confidence Factor (CF): This parameter determines the confidence level used for pruning. Higher values of the confidence factor result in more aggressive pruning, leading to smaller trees.

Subtree Raising Confidence (SubtreeRaising): This parameter controls subtree raising, which is a technique used to prune the tree after it has been built. Increasing this value can lead to more aggressive pruning of subtrees, resulting in smaller trees.

Use Reduced Error Pruning (ReducedErrorPruning): Enabling this option applies reduced error pruning to the tree, which can help prevent overfitting by removing branches that do not improve overall accuracy.

Use MDL Correction (UseMDLCorrection): MDL (Minimum Description Length) correction is used to adjust the decision tree model based on the complexity of the data. Enabling this option can help control the growth of the tree by penalizing complex models.

## B. Describe and explain each of the classification metrics reported by Weka. ---------- <a name="section-2"></a>

In the output of a J48 classifier in Weka, you'll typically encounter several performance metrics. Here's what each of them means:

Correctly Classified Instances: This refers to the number of instances that were classified correctly by the model.

Incorrectly Classified Instances: This refers to the number of instances that were classified incorrectly by the model.

Kappa statistic: This is a statistic that measures the agreement between the observed accuracy and the expected accuracy. It takes into account the possibility of the correct classification occurring by chance.

Mean absolute error: This is the average of the absolute errors between the predicted and actual values.

Root mean squared error: This is the square root of the average of the squared errors between the predicted and actual values.

Relative absolute error: This is the mean absolute error divided by the mean absolute error of the baseline predictor.

Root relative squared error: This is the root mean squared error divided by the root mean squared error of the baseline predictor.

TP Rate (True Positive Rate): This is the ratio of correctly predicted positive instances to all actual positive instances.

FP Rate (False Positive Rate): This is the ratio of incorrectly predicted positive instances to all actual negative instances.

Precision: This is the ratio of correctly predicted positive observations to the total predicted positives. It measures the accuracy of positive predictions.

Recall: This is the ratio of correctly predicted positive observations to all actual positives. It measures the ability of the classifier to find all positive instances.

F-Measure: This is the harmonic mean of precision and recall. It provides a single score that balances both precision and recall.

MCC (Matthews Correlation Coefficient): This is a correlation coefficient between the observed and predicted binary classifications.

ROC Area: This is the area under the receiver operating characteristic (ROC) curve. It provides an aggregate measure of performance across all possible classification thresholds.

PRC Area: This is the area under the precision-recall curve. It provides an aggregate measure of performance across all possible classification thresholds.

Confusion Matrix: This matrix summarizes the actual and predicted classifications. It shows the counts of true positive, true negative, false positive, and false negative predictions, allowing for a detailed analysis of classifier performance.
