#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 00:21:00 2023

@author: bee
"""

import pandas as pd 

# Reading in the data set. read_csv is the function to read in csv files. 
#In double quotes, provide the path and file name. 
data = pd.read_csv("/Users/bee/Desktop/Datasets/dementia_dataset.csv")

print("DATA EXPLORATION")

data.head()
data.tail()
data.describe()

# To check the no. of rows and columns. 
num_rows, num_columns = data.shape

print("Number of rows:", num_rows)
print("Number of columns:", num_columns)

print('\n')  # Insert a blank line


print("DATA CLEANING")

# Get the data types of each column
column_types = data.dtypes

# Separate the columns into numerical and categorical
numerical_columns = column_types[column_types != 'object'].index.tolist()
categorical_columns = column_types[column_types == 'object'].index.tolist()

print("Numerical Columns:", numerical_columns)
print("Categorical Columns:", categorical_columns)

print('\n')  # Insert a blank line

# Change the str of certain columns to categorical using astype()
#At this point, once you convert the structure of the variables, the variable explorer will stop working, idk why

data['Visit'] = data['Visit'].astype('category')
data['SES'] = data['SES'].astype('category')
data['Group'] = data['Group'].astype('category')
data['M/F'] = data['M/F'].astype('category')
data['Hand'] = data['Hand'].astype('category')

print('\n')  # Insert a blank line

data['M/F'] = data['M/F'].replace({'M':1, 'F':2})

data['Hand'] = data['Hand'].replace({'R':1})

data['Group'] = data['Group'].replace({'Nondemented':1, 'Demented':2, 'Converted':3})

print(data)

print(data.dtypes)


#Getting the names of the columsn 

data.columns.values.tolist()

#This is to check if there is any missing data. This dataset for instance has 
#... 21 missing values. We will go ahead and replace them with the median. It 
# is th simplest form of imputation and will do the job for now. 

print("Missing values(Y/N) in each col of the data")

for column in data.columns:
    is_missing = data[column].isnull().any()
    
    if is_missing:
        print(f"The '{column}' column has missing values.")
    else:
        print(f"The '{column}' column does not have missing values.")
        

print('\n')  # Insert a blank line

        
#To check no. of unique factors/levels of a column 

print("To check no. of unique factors/levels of a column")

column_name = 'Group'  

num_levels = data[column_name].nunique()

print(f"The '{column_name}' column has {num_levels} unique factors/levels.")

print('\n')  # Insert a blank line

print("DATA PREPROCESSING")

print("HANDLING MISSING DATA")

# To check for missing values 

for column in data.columns:
    is_missing = data[column].isnull().any()
    
    if is_missing:
        print(f"The '{column}' column has missing values.")
    else:
        print(f"The '{column}' column does not have missing values.")

# #Impute with the mode: If the 'SES' column contains categorical data, 
#you can impute the missing values with the mode, which is the most frequent value in the column. 
#This approach is suitable for categorical variables.

#This code replaces the missing values in the 'SES' column with the mode value using the fillna() function.

data['SES'].fillna(data['SES'].mode()[0], inplace=True)

#This code replaces the missing values in the 'MMSE' column with the median value of the non-missing values.
median_value = data['MMSE'].median()
data['MMSE'].fillna(median_value, inplace=True)

print('\n')  # Insert a blank line

print(" DATA SCALING")

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

#Feautures
X = data.drop(['Group', 'Subject ID', 'MRI ID'], axis =1)

#Target variable
y = data['Group']


#Specify the numerical features
numeric_features = ['MMSE', 'Age', 'EDUC', 'CDR', 'eTIV', 'nWBV', 'ASF']

# Numerical Scaling 

scaler = MinMaxScaler()
X[numeric_features] = scaler.fit_transform(data[numeric_features])

# To verify is the numeric vatiables have been sucessfully scaled 
print(X[numeric_features])

print("End of Numerical scaling")  # Insert a blank line


#Categorical Scaling 
#This line specifies the column names of the categorical features in your dataset. In this case, "SES" and "Sex" are assumed to be categorical columns.

categorical_features = ['Visit','M/F', 'Hand' ,'SES']

#Here, you create an instance of the OneHotEncoder class. This encoder is used to transform categorical features into one-hot encoded representations. 
#The sparse=False argument ensures that the output is a dense matrix, and drop='first' drops the first category for each feature to avoid multicollinearity.

# <summary>Why dropping the first value of each scaled categorical variable is important:</summary>
#
# Dropping the first value of each scaled categorical variable is done to avoid multicollinearity, which is a situation
# where two or more variables are highly correlated and provide redundant information to the model. Here's why dropping
# the first value is important:
#
# - One-Hot Encoding: When we apply one-hot encoding to a categorical variable, we create binary columns for each unique
#   category. For example, if we have a "Color" variable with categories "Red," "Blue," and "Green," after one-hot encoding,
#   we would have three binary columns: "Color_Red," "Color_Blue," and "Color_Green."
#
# - Dummy Variable Trap: The dummy variable trap refers to the scenario where one of the encoded columns can be predicted
#   perfectly using the others. By including all the binary columns derived from a categorical variable, we introduce
#   multicollinearity into the model. This multicollinearity can lead to issues such as unstable model coefficients, inflated
#   standard errors, and difficulty in interpreting the importance of individual features.
#
# - Dropping the First Value: To avoid the dummy variable trap, we drop the first binary column corresponding to a categorical
#   variable. In the example above, we would drop either "Color_Red," "Color_Blue," or "Color_Green." By dropping one of the
#   columns, we ensure that the model can accurately represent all possible categories without introducing redundancy or
#   collinearity.
#
# It's important to note that dropping the first value doesn't result in any loss of information since the omitted category
# can be inferred from the absence of all other categories. Additionally, most machine learning libraries, including scikit-learn,
# automatically handle the dummy variable trap by dropping one column during one-hot encoding. The drop='first' parameter in the
# OneHotEncoder class is used to specify this behavior explicitly.
#
# By dropping the first value of each scaled categorical variable, we ensure that our encoded features are not redundant and
# don't introduce multicollinearity, leading to more reliable and interpretable models.

categorical_encoder = OneHotEncoder(sparse_output=True, drop='first')

#ct = ColumnTransformer([('categorical_encoder', categorical_encoder, categorical_features)], remainder='passthrough'): 
#This line creates a ColumnTransformer object called ct. It allows you to specify different transformations for different columns in your dataset. 
#In this case, you want to apply the categorical_encoder to the categorical_features columns. 
#The remainder='passthrough' argument ensures that the remaining non-categorical columns are passed through without any transformation.

ct = ColumnTransformer([('categorical_encoder', categorical_encoder, categorical_features)], remainder='passthrough')

#X_encoded = ct.fit_transform(X): Finally, you apply the column transformation to your original feature matrix X using the fit_transform method of the ColumnTransformer object. 
#This performs the one-hot encoding transformation on the specified categorical columns and concatenates the transformed columns with the remaining non-categorical columns.
#The resulting X_encoded will be a new feature matrix with the categorical features replaced by their one-hot encoded representations and the non-categorical features unchanged.
#By using the OneHotEncoder and ColumnTransformer in this way, you can easily handle categorical features in your dataset and apply the necessary transformations for further modeling or analysis.

X_encoded = ct.fit_transform(X)

print("Principal Component Analysis")

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)  # Set the number of components as needed
X_pca = pca.fit_transform(X_scaled)

# Create a new DataFrame with the PCA results
pca_data = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])
pca_data['Group'] = data['Group']

# Print the explained variance ratio
print("Explained Variance ratio:")
print(pca.explained_variance_ratio_)

# Visualizing PCA results
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_data['PC1'], pca_data['PC2'], pca_data['PC3'], c=pca_data['Group'])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.title('PCA 3D Scatterplot')
plt.show()

kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(X_pca)
labels = kmeans.labels_

print("Cluster Labels:")
print(labels)

X_train, X_test, y_train, y_test = train_test_split(X_pca, data['Group'], test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predicted Labels:")
print(y_pred)

silhouette = silhouette_score(X_pca, labels)
calinski_harabasz = calinski_harabasz_score(X_pca, labels)
davies_bouldin = davies_bouldin_score(X_pca, labels)

y_pred = model.predict(X_pca)
accuracy = accuracy_score(data['Group'], y_pred)
precision = precision_score(data['Group'], y_pred, average='weighted')
recall = recall_score(data['Group'], y_pred, average='weighted')
f1 = f1_score(data['Group'], y_pred, average='weighted')
confusion = confusion_matrix(data['Group'], y_pred)

print("Clustering Evaluation Metrics:")
print("Silhouette Score:", silhouette)
print("Calinski-Harabasz Index:", calinski_harabasz)
print("Davies-Bouldin Index:", davies_bouldin)

print("\nClassification Evaluation Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion)








































































