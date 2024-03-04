# Random-forest-algorithm
## Description-
This Python program utilizes the pandas library to preprocess a dataset, including handling missing values and encoding categorical variables. It then splits the dataset into training and testing sets, trains a RandomForestClassifier model on the training data, and evaluates its performance on the test data using accuracy and confusion matrix. The model aims to predict the target variable based on the features provided in the dataset, making it a useful tool for classification tasks in data science.
## Explanation-
  -Import necessary libraries for data manipulation and machine learning tasks.
  
  -Load the dataset from a CSV file into a pandas DataFrame.
  
  -Fill missing values in the dataset with the mean of each column.
  
  -Encode categorical variables in the dataset using one-hot encoding.

  -Split the dataset into features (X) and the target variable (y) to be predicted.
  
  -Further split the data into training and testing sets using a 80-20 ratio.
  
  -Initialize a RandomForestClassifier model with 100 trees for classification.
  
  -Train the model on the training data to learn patterns.
  
  -Predict the target variable using the trained model on the test data.
  
  -Calculate the accuracy of the model by comparing predicted values with actual values.
  
  -Print the accuracy score to evaluate model performance.
  
  -Calculate the confusion matrix to further evaluate model performance.
  
  -Print the confusion matrix.
