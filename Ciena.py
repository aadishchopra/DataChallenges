#import required libraries and functions
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import numpy as np
import statistics
from sklearn.impute import KNNImputer
#Import the dataset into python
df=pd.read_csv('TrainingDataset[98].csv')
df.head()

#Exploratory Data Analysis

#1. Check the min , max and percentile distribution
print(df.describe())

#2. Make histograms and check for normal distribution
#pyplot.hist(df['Outcome_M1'])

#3. Outliers check
# Box Plot

#sns.boxplot(df['Outcome_M4'])
'''
For multiple subplots

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
'''


#Check missing data in each column.

def missing_values(df):
    l=[]
    for col in df:
        nulls=df[col].isnull().sum()
        colname=df[col].name
        m=[nulls,colname]
        l.append(m)
    missing_values_df = pd.DataFrame(l, columns=["Nulls", "Colname"])
    return missing_values_df

missing_values_df=missing_values(df.iloc[:,13:558])
# We observe that the above columns have nulls >50% so there is no point in imputing the data.We will just
#remove these features
Remove_Columns=missing_values_df[missing_values_df['Nulls']>50].Colname.to_list()
df=df.drop(Remove_Columns, axis = 1)


#categorical_columns start with cat
# Check for data sanctity (any data entry/capture error).

''' 
1.Need to remove categories having no variation (safe to remove since this is noise)
2.Need to check categories where some categories have unusually high number of variations 
'''

remove_noise_columns=[]
for i in df:
    if i.startswith("Cat_"):
        if len(df[i].unique()) == 1:
            remove_noise_columns.append(i)
    elif i.startswith("Quan_"):
        if len(df[i].unique()) == 1:
            remove_noise_columns.append(i)


df=df.drop(remove_noise_columns,axis=1)


#Data check for quant data
'''
1. Whether the quant data is breaking any data type rules - Can check by summing and forcing nulls to 0
2. No negative values are allowed in the target variable
3. Check for outliers in the quant data
'''

#Point 1
Quan_Data_Type = [sum(df[x].fillna(0)) for x in df if x.startswith('Quan')]
#Point 2
No_Negative_Values = [(df[x].fillna(0)<0).sum() for x in df if x.startswith('Month')]
#Point 3
#creating a function to remove the outliers

#5. Need to impute the missing data

#Imputing quantitative variables by mean of the data_shape
#Imputing categorical varibles by mean

# using median

for i in df:
    if i.startswith("Cat_"):
        df[i].fillna(df[i].median(), inplace=True)
    elif i.startswith("Quan_"):
        df[i].fillna(df[i].median(), inplace=True)
    elif i.startswith("Date"):
        df[i].fillna(df[i].median(), inplace=True)
    elif i.startswith("Outcome_"):
        df[i].fillna(int(df[i].median()), inplace=True)

'''
1. Since the datapoints are not that many wiping off the data using Q3+1.5*IQR will reduce the dataset.
Therefore we can look for alternatives such as transformation 

def remove_outliers(df):
    for x in df:
        if x.startswith('Outcome'):
            Q1 = np.percentile(df[x], 25,method = 'midpoint')
            Q3 = np.percentile(df[x], 75,method='midpoint')
            IQR = Q3 - Q1
            print("Old Shape: ", df.shape)
            print('Removing from '+ x )
            # Upper bound
            upper = np.where(df[x] >= (Q3 + 1.5 * IQR))
            # Lower bound
            lower = np.where(df[x] <= (Q1 - 1.5 * IQR))
            print('Removing %s upper datapoints ',len(upper[0]))
            df.drop(upper[0], inplace=True,)
            df=df.reset_index(drop=True)
            print('Removing %s lower datapoints ', len(lower[0]))
            df.drop(lower[0], inplace=True)
            df=df.reset_index(drop=True)
            print('Total %s datapoints (upper +lower) removed',len(upper[0])+len(lower[0]) )
            print("New Shape: ", df.shape)
remove_outliers(df)

'''
# IQR

#pyplot.scatter(x=df.index,y=df['Outcome_M1'])

#Manual inspection of data


#Split the train and test data
data_shape=df.shape
x_train,x_test,y_train,y_test=train_test_split(df.iloc[:,13:558],df.iloc[:,0:12],train_size=0.8)

# We observe that the above columns have nulls >50% so there is no point in imputing the data.We will just
#remove these features

'''
knn = KNNImputer(n_neighbors=2, add_indicator=True)
knn.fit(x_train)
x_train_t=knn.transform(x_train)

'''

# Dimensionality Reduction

#5 Using Random Forest


classifier = RandomForestRegressor(n_estimators = 600, random_state = 42)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
errors = abs(y_pred - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# Regression

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print(f'Accuracy score without normalization: {metrics.mean_squared_error(y_test, y_pred)}')


## Min-max normalization
sc = MinMaxScaler()
X_train_norm = sc.fit_transform(x_train)
X_test_norm = sc.transform(x_test)
classifier.fit(X_train_norm, y_train)
y_pred = classifier.predict(X_test_norm)
print(f'Accuracy score with min-max normalization: {metrics.mean_squared_error(y_test, y_pred)}')

## Standardization
sc = StandardScaler()
X_train_norm = sc.fit_transform(x_train)
X_test_norm = sc.transform(x_test)
classifier.fit(X_train_norm, y_train)
y_pred = classifier.predict(X_test_norm)
print(f'Accuracy score with standardization: {metrics.mean_squared_error(y_test, y_pred)}')


#The above suggests that there is hardly any performance imporvement by ising min-max or standard
#scaler
