# Project_2

## Contents

* **Stroke Prediction** 
  - Stroke prediction project that leverages a dataset of ### people with ### data points to assertain the outcome of potential strokes for medical predictive outcomes
  - Conducted in-depth analyses of multiple major factors related to brain health to identify and isolate highly correlated elements to better predict stroke risk
  - Explored the impact of lifestyle choices on brain health to help hospitals uncover valuable correlations and potential avenues for stroke mitigation
   

* **Overview**
  - This project developes a pipeline for data entry of individual records to predict the potential adverse outcome of a stroke.
  - While this dataset could be helpful at predicting the potential for having a stroke, it has its limitations and pitfalls in its scope of predictive potential

* **Example Usage**: Phyisicians in a hospital or clinical setting could potentially use a pipeline like this given the easily obtained parameters to measure the potential negative outcome or risk of the adverse outcome of having a stroke 

* **Getting Started**
  - loading the data

  - cleaning the data:
    - dropna
    - value_counts
        - data is imbalanced 
            - create synthetic balance using SMOTE
    - convert objects to categorical variables
    - encode features
    - Scale the data
        - standard scaler 
    - travis-ci results
    - mailing list

  - analyzing the data:
    - decision tree
    - PCA
    - Random Forest
        - Max depth 
        - confusion matrix
    - K Neighbors
    - Cat Boost 
    - XGBoost
  

* **Colophon**
  - Credits --  original code written by: Sirisha Mandava, Jeff Boczkaja, Mohamed Altoobli, Jesse Kranyak
  - inspiration for project-- 
    -  https://www.kaggle.com/code/thomaskonstantin/analyzing-and-modeling-stroke-data
    -  https://www.stat.cmu.edu/capstoneresearch/spring2021/315files/team16.html

* **Data**

 - this data set is public and provided by kaggle. 
