import numpy as np
import pandas as pd
import math
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.ensemble import GradientBoostingRegressor



#Check for NaN in data
def inspect_missing_data():
    print("\nInspecting the Missing Data:")
    columnList = (list(data.columns))[:-1]
    i = 0
    for col in columnList:
        print("There are", pd.isnull(X_test[:,i]).sum(),"nan values in", col, "column")
        i = i + 1


#Investigating Categorical Items
def inspect_categorical_data():
    print("\nInvestigating the Categorical Features")
    cat_item_indexes = [1,3,5,6,8]
    columnList = (list(data.columns))[:-1]
    for i in cat_item_indexes:
        some_set = set_of_category_items(X[:,i])
        print("There are", len(some_set), "unique items in the ", columnList[i], "set")
        print(some_set,"\n\n")

def set_of_category_items(full_list):
    some_set = set()
    for j in full_list:
        some_set.add(j)
    return some_set

def find_country_once(X, X_test):
    X = X[:,3]
    X_test = X_test[:,3]
    countries = {}
    for c in X:
        if c in countries:
            countries[c] = (countries[c]+1)
        else:
            countries[c] = 1
    for c in X_test:
        if c in countries:
            countries[c] = (countries[c]+1)
        else:
            countries[c] = 1
    for c in countries:
        if countries[c] == 1:
            print(c)




data = pd.read_csv('training.csv')
test_data = pd.read_csv('test.csv')

print("Initial shape of data Matrix: ",data.shape)
print("Initial shape of test Matrix: ",test_data.shape)


#DROP INSTANCE COLUMN as irrelevant
data = data.drop("Instance", axis=1)
test_data = test_data.drop("Instance", axis=1)

#Print column headers
print(data.columns)
print(test_data.columns)


#Load data into X and y matrices
X = data.iloc[:,:-1].values
y = data.iloc[:, -1].values
X_test = test_data.iloc[:,:-1].values

#List new headers of columns
print("Headers of columns:", list(data.columns))
print("New size of X matrix:", X.shape)
print("New size of y array:", y.shape)

#Inspect head of data
print(data.head)

#Replace Nan in numerical columns with mean value
numerical_columns = [0,2,4,7,9] #Year of Record, Age

inspect_missing_data()
for i in numerical_columns:
    imputer = SimpleImputer(missing_values = np.nan, strategy="mean")
    imputer.fit(X[:,i].reshape(-1,1))
    X[:,i] = imputer.transform(X[:,i].reshape(-1,1))[:,0]
    X_test[:,i] = imputer.transform(X_test[:,i].reshape(-1,1))[:,0]

#Check for NaN
inspect_missing_data()

#Inspect categorical data
inspect_categorical_data()

#List countries that appear once
print("\nCountries that only appear once:")
find_country_once(X, X_test)


#Convert nan and 0 values to Unknown
cat_item_indexes = [1,3,5,6,8]
for i in cat_item_indexes:
    temp = X[:,i]
    nan_indexes = pd.isnull(X[:,i])
    temp[nan_indexes] = "unknown"
    temp[temp=="0"]="unknown"
    temp[temp=="Unknown"] = "unknown"
    X[:,i] = temp
    temp = X_test[:,i]
    nan_indexes = pd.isnull(X_test[:,i])
    temp[nan_indexes] = "unknown"
    temp[temp=="0"]="unknown"
    temp[temp=="Unknown"] = "unknown"
    X_test[:,i] = temp

#Encode categorical data
print("Encoding data..")
encoder_t = CatBoostEncoder(cols=cat_item_indexes)
X = encoder_t.fit_transform(X,y)
X_test = encoder_t.transform(X_test)
X_test = X_test.astype(float)
X_test = X_test.iloc[:,:].values
X = X.astype(float)
X = X.iloc[:,:].values

#Scale data
print("Scaling..")
sc = RobustScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)

#Fit model - n_estimators, max_depth  & min_samples_split at current values will take a long time to run.
#Reducing these values will reduce the RMSE by a small margin, but testing will be a lot faster.
print("\nBeginning Gradient Boosting Regression.")
gbrReg = GradientBoostingRegressor(n_estimators=3000, max_depth=6, learning_rate=0.01, min_samples_split = 3)
gbrReg.fit(X,y)
gbrRes = gbrReg.predict(X_test)
np.savetxt("gbrResults.txt", gbrRes, fmt= "%f",newline='\n')
