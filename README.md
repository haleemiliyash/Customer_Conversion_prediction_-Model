# Customer_Conversion_prediction_-Model
## Problem Statement
In the insurance industry, acquiring new customers and converting leads into sales is
crucial for business growth. The dataset provided contains information about a series
of marketing calls made to potential customers by an insurance company. The goal is
to predict whether a customer will subscribe to an insurance policy based on various
attributes of the customer and details of the marketing interactions.

# Data Set:
1. age: Age of the customer.
2. job: Type of job the customer holds.
3. marital: Marital status of the customer.
4. education_qual: Educational qualification of the customer.
5. call_type: Type of marketing call.
6. day: Day of the month when the call was made.
7. mon: Month when the call was made.
8. dur: Duration of the call in seconds.
9. num_calls: Number of calls made to the customer before this interaction.
10.prev_outcome: Outcome of the previous marketing campaign.
11. y: Whether the customer subscribed to the insurance policy (target variable).

    
# Data -https://raw.githubusercontent.com/GuviMentor88/Training-Datasets/main/insurance_dataset.csv

## Libraries Used:

* Streamlit
* Pandas
* Numpy 
* Random forest Classifier
* Json

## Approach:
1. Data Preprocessing: Clean the data, handle missing values, Scale the dataset with SMOTEEN Oversampling method
Normalize Features(), and Lable encoder for categorical features.

2. Exploratory Data Analysis (EDA): Understand the distribution of features,
identify patterns, and explore relationships between features and the target
variable.

3. Dataset Balancing: The target feature are Not Balanced . The imbalance of
the target variable are processed by oversampling method of SMOTTEN technique.

4. Model Building: Train various machine learning models to predict the target
variable  like Logistic Regression, KNN Classification,XGB Classification,Descision Tree, Random Forest classifier
● Find the correlation between features by using Heat map
● Split the dataset into training and testing/validation sets.
● Train and evaluate all the different classification models, such as
like Logistic Regression, KNN Classification,XGB Classification,Descision Tree, Random Forest classifierand Extra tree classifier.
● Train the clustering model: Customer segmentation based on different
groups with Kmean algorithum used.
● Hyperparameter Tuning: Optimize model hyperparameters using
techniques such as cross-validation and grid search or Random search
to find the best-performing model.
