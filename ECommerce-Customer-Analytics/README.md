# E-Commerce Customer Analytics

This project showcases a series of data analysis and machine learning models applied to an e-commerce dataset. The primary goals are to predict customer churn and to segment customers based on their behavior and characteristics. This repository is intended to demonstrate skills in data preprocessing, feature engineering, model building, and evaluation for both supervised and unsupervised learning tasks.

## Table of Contents
- [Project Objective](#project-objective)
- [Dataset](#dataset)
- [Analyses and Models](#analyses-and-models)
  - [Customer Churn Prediction (Supervised Learning)](#customer-churn-prediction-supervised-learning)
  - [Customer Segmentation (Unsupervised Learning)](#customer-segmentation-unsupervised-learning)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)

## Project Objective

The main objectives of this project are:
1.  **To predict customer churn:** By building and comparing various classification models, we can identify customers who are likely to stop using the e-commerce service. This allows for proactive retention strategies.
2.  **To segment customers:** Using clustering algorithms, we can group customers into meaningful segments. This helps in understanding the customer base better and enables targeted marketing campaigns.

## Dataset

The project uses the `ECommerce_Churn_Data.csv` dataset. It contains various customer attributes, including:
- Demographics: `Gender`, `MaritalStatus`, `CityTier`
- Behavioral: `Tenure`, `PreferredLoginDevice`, `HourSpendOnApp`, `NumberOfDeviceRegistered`, `PreferedOrderCat`, `SatisfactionScore`, `Complain`
- Transactional: `OrderAmountHikeFromlastYear`, `CouponUsed`, `OrderCount`, `DaySinceLastOrder`, `CashbackAmount`
- Target Variable: `Churn` (1 for churned, 0 for not churned)

## Analyses and Models

This repository includes a collection of Jupyter notebooks, each exploring a different machine learning model or analysis technique.

### Customer Churn Prediction (Supervised Learning)

A variety of classification algorithms were implemented to predict customer churn. Each model was trained and evaluated on the dataset, with a focus on performance metrics like Accuracy, Precision, Recall, F1-Score, and ROC AUC.

- **Logistic Regression:** A baseline model for binary classification.
- **K-Nearest Neighbors (KNN):** A non-parametric algorithm that classifies based on the majority class of its nearest neighbors.
- **Decision Trees:** A simple yet powerful model that creates a tree-like structure for decision making.
- **Random Forests:** An ensemble method that builds multiple decision trees to improve predictive accuracy and control over-fitting.
- **Support Vector Machines (SVM):** A model that finds an optimal hyperplane to separate classes.
- **Gradient Boosting Models:**
  - **XGBoost:** A highly efficient and effective implementation of gradient boosting.
  - **LightGBM:** A fast, distributed, high-performance gradient boosting framework.
  - **CatBoost:** A gradient boosting library that works well with categorical data.
- **Neural Networks:** A deep learning model built with Keras/TensorFlow for churn prediction.

### Customer Segmentation (Unsupervised Learning)

To understand the customer base better, several clustering algorithms were used to group customers into distinct segments.

- **K-Means Clustering:** An iterative algorithm that partitions the dataset into a pre-determined number of clusters.
- **Hierarchical Clustering:** A method that creates a tree of clusters (dendrogram).
- **DBSCAN:** A density-based clustering algorithm that can find arbitrarily shaped clusters and identify noise points.

## Key Findings

- **Churn Prediction:** The gradient boosting models (XGBoost, LightGBM, CatBoost) and Random Forest generally provided the highest accuracy and ROC AUC scores, indicating their effectiveness in predicting customer churn. Key predictors of churn often include `Tenure`, `Complain`, and `DaySinceLastOrder`.
- **Customer Segmentation:** The clustering analyses revealed distinct customer personas, such as:
  - **Loyal High-Value Customers:** High tenure, frequent orders, and high cashback amounts.
  - **At-Risk Customers:** Low satisfaction scores, high complaint rates, and low tenure.
  - **New Customers:** Low tenure and order count.

These insights can be used to tailor marketing efforts, improve customer service, and develop targeted retention campaigns.

## Technologies Used

- **Python 3**
- **Jupyter Notebook**
- **Data Manipulation and Analysis:**
  - **Pandas**
  - **NumPy**
- **Data Visualization:**
  - **Matplotlib**
  - **Seaborn**
- **Machine Learning:**
  - **Scikit-learn** (for Logistic Regression, KNN, Decision Trees, SVM, Clustering, etc.)
  - **XGBoost**
  - **LightGBM**
  - **CatBoost**
  - **TensorFlow / Keras** (for Neural Networks)

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd Data-ML-Portfolio/ECommerce-Customer-Analytics
    ```
3.  **Install dependencies:**
    It is recommended to create a virtual environment first.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file should be created to list all necessary packages.)*

4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
5.  Open and run the individual `*.ipynb` files to see the analysis and model implementation for each technique.
