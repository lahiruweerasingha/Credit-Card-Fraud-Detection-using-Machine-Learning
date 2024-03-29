# **Credit Card Fraud Detection using Machine Learning**

# **Description:**
This project aims to develop a machine learning model for detecting fraudulent credit card transactions. Fraudulent transactions pose a significant threat to credit card companies and consumers, making it essential to identify and prevent them accurately. The dataset used contains credit card transactions made by European cardholders in September 2013, where the positive class (frauds) constitutes only 0.172% of all transactions, leading to a highly imbalanced dataset.

# **Dataset Obtained from:**
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud


### Technical Stack:
1. **Python Libraries:**
   - pandas
   - numpy
   - scikit-learn
   - imbalanced-learn (SMOTE for handling imbalanced dataset)
   - matplotlib
   - seaborn
   - joblib

2. **Modeling Techniques:**
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier

### Workflow Overview:

1. **Importing Necessary Libraries:** 
   Importing required Python libraries for data manipulation, preprocessing, model training, evaluation, and visualization.

2. **Load the Dataset:** 
   Loading the credit card transaction dataset containing information on transaction amounts, time, and PCA-transformed features.

3. **Data Preprocessing & Exploratory Data Analysis:** 
   Performing data preprocessing steps such as checking for missing values, scaling features, dropping unnecessary columns, handling duplicate entries, and exploring class distribution through visualizations.

4. **Data Splitting for Model Training and Evaluation:** 
   Splitting the dataset into training and testing sets for model training and evaluation.

5. **Handling Imbalanced Dataset:** 
   Addressing the class imbalance issue through both undersampling and oversampling techniques.

   - **Undersampling:** Reducing the majority class instances to balance the dataset.
   - **Oversampling:** Generating synthetic samples for the minority class to balance the dataset.

6. **Model Training and Evaluation:** 
   Training logistic regression, decision tree, and random forest classifiers on both undersampled and oversampled datasets. Evaluating the models based on accuracy, precision, recall, and F1 score.

7. **Model Selection:** 
   Comparing the performance of models and selecting the best-performing model for predictions.

8. **Save the Model:** 
   Saving the trained model for future use. An example prediction is demonstrated using the saved model.

### Conclusion:
The Random Forest Classifier with oversampling emerges as the best-performing model for credit card fraud detection. This project showcases the importance of handling imbalanced datasets and demonstrates the effectiveness of machine learning techniques in mitigating credit card fraud risks.

