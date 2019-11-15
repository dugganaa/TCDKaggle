import pandas as pd
import numpy as np
import blogPreprocessing as bp
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Returns list of Columns that have all Nans - uses columnNumber from original Data
#Must be altered if planning on calling after removing columns
def inspect_missing_data(X):
    print("Inspecting the Missing Data:")
    listOfAllNans = []
    columnList = (list(data.columns))[:-1]
    i = 0
    for col in columnList:
        backNCount = 0
        notProvidedCount = 0
        backN = "\\N"
        notProvided = "Not Provided"
        for j in X[:,i]:
            if j == backN:
                backNCount = backNCount + 1
            if j == notProvided:
                notProvidedCount = notProvidedCount + 1
        print("There are", (pd.isnull(X[:,i]).sum() + backNCount + notProvidedCount),"nan values in", col, "column")
        i = i + 1
        if X[:,i].size == backNCount:
            listOfAllNans.append(col)
    return listOfAllNans

#Returns list of Columns that have all Nans - uses columnNumber from original Data
#Must be altered if planning on calling after removing columns
def inspect_timezone(X):
    usertimezone = X[:,(list(data.columns).index("user_timezone"))]
    timezoneip = X[:,(list(data.columns).index("timezone_by_ip"))]
    for i in range(0, usertimezone.size):
        if usertimezone[i] != "\\N" and usertimezone[i] != "Not provided":
            if usertimezone[i] == timezoneip[i]:
                print("Same value for user_timezone and time_zone_ip")
            else:
                print("Usertimezone val: ", usertimezone[i])
                print("timezoneip val: ", timezoneip[i])

#TODO:
def only_one_val_in_columns(X, colList):
    j = 0
    for col in colList:
        values = {}
        for i in X[:,j]:
            if i not in values:
                values[i] = 1
            else:
                values[i] = values[i] + 1
        print("There are ", len(values), " unique items in ", col)
        j = j + 1

data = pd.read_csv('training.csv')
data_test = pd.read_csv('test.csv')
print("Matrix shape: ", data.shape)
print("Data columns: ", data.columns)


#Training Data
blogX = data.loc[data['organization_id'] == 8]
blogY = blogX['set_clicked']
blogX = blogX.drop(columns=['local_time_of_request', 'local_hour_of_request', 'organization_id', 'response_delivered','user_id','session_id',
    'document_language_provided','year_published','number_of_authors','first_author_id','num_pubs_by_first_author','app_version','app_lang',
    'user_os','user_os_version','user_java_version','user_timezone', 'application_type', 'item_type','abstract_detected_language'
    ,'number_of_recs_in_set','set_clicked'])

#Test Data
blogX_test = data_test.loc[data_test['organization_id'] == 8]
blogX_test = blogX_test.drop(columns=['local_time_of_request', 'local_hour_of_request', 'organization_id', 'response_delivered','user_id','session_id',
    'document_language_provided','year_published','number_of_authors','first_author_id','num_pubs_by_first_author','app_version','app_lang',
    'user_os','user_os_version','user_java_version','user_timezone', 'application_type', 'item_type','abstract_detected_language'
    ,'number_of_recs_in_set','set_clicked'])

print("\nInspect missing Joeran Blog Data:")
#allNans = inspect_missing_data(blogX.values)
#print("Columns that have all Nans: ", allNans)
print("All columnns: ", blogX.columns)
#print("Inspecting timezone: ")
#inspect_timezone(blogX)
#print("Checking for single value columns: ")
#only_one_val_in_columns(blogX)
print("Joeran Dataset Size: ", blogX.values.shape)


'''
Remove Columns:
user_id                             | Nan
session_id                          | Nan
document_language_provided          | Nan
year_published                      | Nan
number_of_authors                   | Nan
first_authord_id                    | Nan
num_pubs_by_first_author            | Nan
app_version                         | Nan
app_lang                            | Nan
user_os                             | Nan
user_os_version                     | Nan
user_java_version                   | Nan
user_timezone                       | Nan
application_type                    | Only 'blog' with some 0s
item_type                           | Only 'article' with some Nans
abstrat_detected_language           | Only 'en' with some Nans
local_time_of_request               | Mostly Nans - Can infer local time from serverside time of request
local_hour_of_request               | Mostly Nans - Can infer local time from serverside time of request
response_delivered                  | Dependent on hour of request and processing time
time_recs_received                  | Not in test set
time_recs_displayed                 | Not in test set
time_recs_viewed                    | Not in test set
number_of_recs_in_set               | Not in test set


Columns to work on:
recommendation_set_id               | CatEncoder
query_identifier                    | CatEncoder
query_word_count                    | StandardScaler
query_char_count                    | StandardScaler
query_detected_language             | OneHotEncoder
query_document_id                   | CatEncoder
abstract_word_count                 | StandardScaler
abstract_char_count                 | StandardScaler
request_received                    | Day - DayZero -> StandardScaler
hour_request_received               | StandardScaler
rec_processing_time                 | StandardScaler
country_by_ip                       | Get average timezone value associated with each country (Mode). Set all timezone strings/'\n's to that timezone value
timezone_by_ip                      | Set to numerical timezone -> standardscaler
recommendation_algorithm_id_used    | CatEncoder
algorithm_class                     | LabelEncode -> OneHotEncode
cbf_parser                          | LabelEncode -> OneHotEncode
search_title                        | LabelEncode
search_keywords                     | LabelEncode
search_abstract                     | LabelEncode
clicks                              | Binary -> LabelEncode
ctr                                 | Binary -> LabelEncode
set_clicked                         | Target 
'''


listOfBlogXCols = list(blogX.columns)
cat_cols = ['recommendation_set_id', 'query_identifier', 'query_detected_language', 'query_document_id',  'timezone_by_ip', 'recommendation_algorithm_id_used', 'algorithm_class',
            'cbf_parser', 'search_title', 'search_keywords', 'search_abstract', 'clicks', 'ctr']
num_cols = ['query_word_count', 'query_char_count', 'abstract_word_count', 'abstract_char_count', 'request_received', 'hour_request_received', 'rec_processing_time', 'timezone_by_ip']
ss_cols = ['query_word_count', 'query_char_count', 'abstract_word_count', 'abstract_char_count', 'hour_request_received', 'rec_processing_time', 'timezone_by_ip', 'request_received']
le_cols = ['query_detected_language', 'algorithm_class', 'cbf_parser', 'search_title', 'search_keywords', 'search_abstract', 'clicks', 'ctr']
ohe_cols = ['algorithm_class', 'cbf_parser', 'query_detected_language']
ce_cols = ['recommendation_set_id', 'query_identifier','query_document_id', 'recommendation_algorithm_id_used']
date_cols = ['request_received']
sb_cols = ['clicks', 'ctr']
SB_VAL = 0


only_one_val_in_columns(blogX.values, listOfBlogXCols)
print("\nNew size: ", blogX.shape)


(blogX, blogX_test) = bp.remove_time_from_date(blogX, blogX_test, date_cols)
firstDate = blogX['request_received'].iloc[0]
dayZero = datetime.strptime(firstDate, '%d/%m/%Y')
(blogX, blogX_test) = bp.date_to_int(blogX, blogX_test, date_cols, dayZero)
(blogX, blogX_test) = bp.replace_nan_with_unknown(blogX, blogX_test, cat_cols)
(blogX, blogX_test) = bp.replace_nan_with_mean(blogX, blogX_test, num_cols)
(blogX, blogX_test) = bp.set_binary(blogX, blogX_test, sb_cols, SB_VAL)
(blogX, blogX_test) = bp.cat_encode(blogX, blogX_test, ce_cols,blogY)
(blogX, blogX_test) = bp.scale(blogX, blogX_test, ss_cols)
(blogX, blogX_test) = bp.label_encode(blogX, blogX_test, le_cols)
(blogX, blogX_test) = bp.one_hot_encode(blogX, blogX_test, ohe_cols)
print("End of preprocessing...")


print("Starting Logistic Regression..")
clf = LogisticRegression(solver='lbfgs')
clf.fit(blogX, blogY)
res = clf.predict(blogX_test)
np.savetxt("results.txt", res, fmt= "%f",newline='\n')
