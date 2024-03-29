# Project_2

![cover Project conterbuter](/image/cover.png)

## Table of Contents
1. [Project Purpose / Description](#project-purpose)
2. [What do the metrics measure?](#metrics)
3. [Main Project](#main)
   - [Importing Data](#import)
   - [Analyzing and Exploring our Data](#Exploring)
   - [Encoding our data for use in machine learning](#Encoding)
   - [Choose scaling method](#scaling)
   - [Decision Tree](#Decision)
   - [Random Forest](#Random)
   - [K Nearest Neighbors](#Nearest)
   - [Random Forest](#Random)
4. [Conclusion](#Conclusion)
5. [References](#references)
6. [Directory Structure](#directory)






<a name="project-purpose"></a>
## Project Purpose / Description

Utilizing the Stroke Prediction Dataset from Kaggle we set out to make a machine learning program that will be able to accurately predict whether or not someone will have a stroke. We try out different models that provided us with varying results. We show our results using a few different metrics including balanced accuracy score, F1 scores, precision, and recall.

<a name="metrics"></a>
What do the metrics measure?

### Precision

Precision measures the accuracy of positive predictions. It is the ratio of true positive predictions to the total number of positive predictions made. In other words, it answers the question, "Of all the instances the model predicted as positive, how many are actually positive?" Precision is particularly important in scenarios where the cost of a false positive is high.

Formula: Precision = True Positives / (True Positives + False Positives)

### Recall

Recall, also known as sensitivity or true positive rate, measures the ability of a model to find all the relevant cases within a dataset. It is the ratio of true positive predictions to the total number of actual positives. Recall answers the question, "Of all the actual positives, how many did the model successfully identify?" Recall is crucial in situations where missing a positive instance is costly.

Formula: Recall = True Positives / (True Positives + False Negatives)

### F1 Score

The F1 Score is the mean of precision and recall. It provides a single metric that balances both the precision and recall of a classification model, which is particularly useful when you want to compare two or more models. The F1 Score is especially valuable when the distribution of class labels is imbalanced. A high F1 Score indicates that the model has low false positives and low false negatives, so it's correctly identifying real positives and negatives.

Formula: F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

### Balanced Accuracy Score

Balanced Accuracy Score is defined as the average of recall obtained on each class, meaning it considers both the true positive rate and the true negative rate. It calculates the accuracy of the model by taking into account the balance between classes. For a binary classification problem, it would be the average of the proportion of correctly predicted positive observations to the total positive observations and the proportion of correctly predicted negative observations to the total negative observations.

Formula: Balanced Accuracy Score = (1/2) * ((TP / (TP + FN)) + (TN / (TN + FP)))




<a name="main"></a>
# Main Project


<a name="import"></a>
## 1. Importing Data
data source 

<p align="center">
  <img src="/image/importing-data.png" alt="importing data" width="800" >
</p>


<a name="Exploring"></a>
## 2. Analyzing and Exploring our Data


<p align="center">
  <img src="/image/exploring-data.png" alt="exploring data" width="600" >
</p>

the sample of what we used to explor data

<p align="center">
  <img src="/image/sample-data.png" alt="sample of data" width="800" >
</p>


<p align="center">
  <img src="/image/corr.png" alt="correlation graph heat map " width="600" >
</p>



<a name="Encoding"></a>
## 3. Encoding our data for use in machine learning

In machine learning, encoding data is essential for preparing categorical variables to be used as input in algorithms. Since most machine learning models require numerical data, categorical variables such as gender, smoking status, or work type need to be encoded into numerical form. This process ensures that the model can effectively interpret and learn from these features, enabling it to make accurate predictions or classifications based on the input data.


Create synthetic balance in the dataset using SMOTE
Due to the imbalance in our dataset we utilize SMOTE and SMOTENC to create synthetic data to improve the outcomes of our machine learning models.

<p align="center">
  <img src="/image/smote-code.png" alt="smote code " width="800" >
</p>

<a name="scaling"></a>
## 4. Choose scaling method

### Normalization 
rescales the features to a fixed range, usually 0 to 1.

Advantages:

Useful when you need to bound your values between a specific range.
Maintains the original distribution without distorting differences in the ranges of values.
Disadvantages:

If your data contains outliers, normalization can squash the "normal" data into a small portion of the range, reducing the algorithm's ability to learn from it.


### Standardization 
rescales data so that it has a mean of 0 and a standard deviation of 1.

Advantages:

Standardization does not bound values to a specific range, which might be useful for certain algorithms that assume no specific range.
More robust to outliers compared to normalization.
Disadvantages:

The resulting distribution will have a mean of 0 and a standard deviation of 1, but it might not be suitable for algorithms that expect input data to be within a bounded range.


<a name="Decision"></a>
## 5. Decision Tree

A decision tree is a hierarchical model that helps in making decisions by mapping out possible outcomes based on different conditions. It's a visual representation where each branch represents a decision based on features in the data, ultimately leading to a prediction or classification.



<a name="Random"></a>
## 6. Random Forest

A Random Forest is a machine learning method used in both classification and regression tasks. It operates by constructing a multitude of decision trees during training time and outputs the mode or average prediction of the individual trees.


<a name="Nearest"></a>
## 7. K Nearest Neighbors

The k-nearest neighbors algorithm predicts the label of a data point based on the labels of its 'k' closest neighbors in the dataset. To classify a new instance, KNN calculates the distance between the instance and all points in the training set, identifies the 'k' nearest points, and then uses a majority vote among these neighbors to determine the instance's label. For regression tasks, it averages the values of these neighbors instead.

<a name="Conclusion"></a>
## Conclusion

The initial analysis of the dataset revealed a significant imbalance, raising concerns about data leakage potential. To mitigate this, we experimented with both SMOTE and SMOTENC for oversampling, with SMOTE demonstrating greater performance in addressing the imbalance.

Upon evaluating various machine learning models for classification purposes, it was observed that prior to tuning the models did not exhibit strong predictive capabilities. However post-tuning improvements were notable, particularly in terms of balanced accuracy scores. Examining other projects that used our dataset had similar findings. An interesting discovery during our investigation was that datasets incorporating bloodwork data tend to yield more accurate stroke predictions. This suggests that lifestyle-based predictive models might best serve as preliminary tools for healthcare professionals, guiding at-risk patients towards more definitive bloodwork analyses.

Despite the challenges presented by lifestyle data, the Random Forest Classifier was the standout model upon tuning, specifically when adjusted to the optimal max depth. This model achieved a balanced accuracy score of 80%, marking it as the most effective among the classifiers we tested for predicting stroke potential. The Random Forest Classifier with an appropriate max depth is what we would recommended as a tool for stroke prediction, emphasizing the model's utility in clinical settings for early stroke risk assessment.

<p align="center">
  <img src="/image/model-performance.png" alt="model performance " width="800" >
</p>

<a name="references"></a>
## 5. References

Credits -  original code written by: Sirisha Mandava, Jeff Boczkaja, Mohamed Altoobli, Jesse Kranyak

> [Our dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) comes from kaggle 

> [inspiration 1  for project](https://www.kaggle.com/code/thomaskonstantin/analyzing-and-modeling-stroke-data) comes from kaggle 

> [inspiration 2  for project](https://www.stat.cmu.edu/capstoneresearch/spring2021/315files/team16.html) comes from kaggle 




<a name="directory"></a>
## 6. Directory Structure 
- image/
  - Contain images used in the project
- old work/
  - contain files which we test and try
- Stroke Prediction Project.ipynb
  -our main file
-healthcare-dataset-stroke-data.csv
  -our main data set
-Presentation Project 2.pdf
  -our Presentation file
- README.md
  -You are Here 
