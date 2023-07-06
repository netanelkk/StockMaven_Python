import yfinance as yf

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

# from pandas_datareader import DataReader
# import pandas_datareader as dr

from datetime import datetime

import warnings

warnings.filterwarnings("ignore")

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
# %matplotlib inline

# For reading stock data from yahoo
# from pandas_datareader.data import DataReader
import yfinance as yf

# For time stamps
from datetime import datetime

start_date = datetime(2000, 1, 1)
end_date = datetime(2023, 5, 19)

stocks_list = [
    'AAPL', 'META', 'MSFT', 'GOOG', 'NKE',
    'NFLX', 'ADBE', 'ZM', 'NVDA', 'HUBS',
    'LH', 'ET', 'KBR', 'WMT', 'SYK',
    'DE', 'APLS', 'HD', 'JBL', 'PSTG',
    'PANW', 'SQ', 'PPL', 'ACGL', 'DDOG',
    'GOOGL', 'ALV.DE', 'PZZA', 'STEM', 'GE',
    'GM', 'SITM', 'BOX', 'SGEN', 'CTVA',
    'DT', 'PLNT', 'GTLS', 'PRVA', 'DG.PA',
    'PINS', 'CVE', 'VMC', 'GLOB', 'CPA',
    'FTI', 'AEM', 'WCN'
]
for stock in stocks_list:
    globals()[stock] = yf.download(stock, start_date, end_date)

company_list = [
    'AAPL', 'META', 'MSFT', 'GOOG', 'NKE',
    'NFLX', 'ADBE', 'ZM', 'NVDA', 'HUBS',
    'LH', 'ET', 'KBR', 'WMT', 'SYK',
    'DE', 'APLS', 'HD', 'JBL', 'PSTG',
    'PANW', 'SQ', 'PPL', 'ACGL', 'DDOG',
    'GOOGL', 'ALV.DE', 'PZZA', 'STEM', 'GE',
    'GM', 'SITM', 'BOX', 'SGEN', 'CTVA',
    'DT', 'PLNT', 'GTLS', 'PRVA', 'DG.PA',
    'PINS', 'CVE', 'VMC', 'GLOB', 'CPA',
    'FTI', 'AEM', 'WCN'
]

company_name = [
    'Apple Inc.', 'Meta', 'Microsoft',
    'Google', 'Nike', 'Netflix',
    'Adobe', 'Zoom Inc.', 'Nvidia',
    'HubSpot', 'Laboratory Corp', 'Energy Transfer',
    'KBR Inc', 'Walmart Inc.', 'Stryker Corp',
    'Deere & Co', 'Apellis', 'Home Depot Inc.',
    'Jabil Inc.', 'Pure Storage', 'Palo Alto',
    'Block Inc.', 'PPL Corp', 'Arch Capital',
    'Datadog Inc.', 'Alphabet Inc.', 'Allianz SE',
    "Papa John's", 'Stem Inc.', 'General Electric',
    'General Motors', 'SiTime Corp.', 'Box Inc.',
    'Seagen Inc.', 'Corteva Inc.', 'Dynatrace Inc.',
    'Planet Fitness', 'Chart Industries', 'Privia Health',
    'Vinci S.A', 'Pinterest', 'Cenovus Energy',
    'Vulcan Materials', 'Globant SA', 'Copa Holdings',
    'TechnipFMC', 'Agnico Eagle', 'Waste Conn.'
]

df_list = []
for company, com_name in zip(company_list, company_name):
    company_df = globals()[company]  # retrieve the corresponding dataframe
    company_df["company_name"] = com_name  # assign the com_name variable to the company_name column of the dataframe
    df_list.append(company_df)

df = pd.concat(df_list, axis=0)
df.tail(10)

import json


def save_volume_min_max_to_json(df, filename):
    grouped = df.groupby('company_name')

    volume_min_max_dict = {}

    for company, group in grouped:
        volume_min_max_dict[company] = {
            'min_volume': float(group['Volume'].min()),
            'max_volume': float(group['Volume'].max())
        }

    with open(filename, 'w') as f:
        json.dump(volume_min_max_dict, f)


save_volume_min_max_to_json(df, 'stocks_volume_min_max.json')

df.info

snoppi_stocks = ['^GSPC']

import yfinance as yf

df_snopi = yf.download(snoppi_stocks, start_date, end_date, group_by='column')
df_snopi
# df_snopi.reset_index(level=0, inplace=True)

vix_stock = ['^VIX']

import yfinance as yf

df_vix = yf.download(vix_stock, start_date, end_date, group_by='column')
df_vix

df_vix = df_vix.sort_values(["Date"], ascending=[False]).reset_index(drop=False)
df_snopi = df_snopi.sort_values(["Date"], ascending=[False]).reset_index(drop=False)

# df_vix.reset_index(level=0, inplace=True)
for i in df_snopi.index[1:]:
    df_snopi.at[i, "GSPC_change_in_day_Adj"] = ((df_snopi.at[i - 1, "Adj Close"] - df_snopi.at[i - 1, "Open"]) /
                                                df_snopi.at[i - 1, "Adj Close"]) * 100
    df_snopi.at[i, "GSPC_change_in_day"] = ((df_snopi.at[i - 1, "Close"] - df_snopi.at[i - 1, "Open"]) / df_snopi.at[
        i - 1, "Adj Close"]) * 100
    df_snopi.at[i, "GSPC_change_in_day_open_high"] = ((df_snopi.at[i - 1, "Open"] - df_snopi.at[i - 1, "High"]) /
                                                      df_snopi.at[i - 1, "Adj Close"]) * 100
    df_snopi.at[i, "GSPC_change_in_day_open_low"] = ((df_snopi.at[i - 1, "Open"] - df_snopi.at[i - 1, "Low"]) /
                                                     df_snopi.at[i - 1, "Adj Close"]) * 100
    df_snopi.at[i, "GSPC_change_in_day_high_low"] = ((df_snopi.at[i - 1, "High"] - df_snopi.at[i - 1, "Low"]) /
                                                     df_snopi.at[i - 1, "Adj Close"]) * 100

df_snopi.rename(columns={'Open': 'Open_GSPC', 'High': 'High_GSPC', 'Low': 'Low_GSPC', 'Close': 'Close_GSPC',
                         'Adj Close': 'Adj Close_GSPC', 'Volume': 'Volume_GSPC'}, inplace=True)

df_snopi

for i in df_vix.index[1:]:
    df_vix.at[i, "VIX_change_in_day_Adj"] = ((df_vix.at[i - 1, "Adj Close"] - df_vix.at[i - 1, "Open"]) / df_vix.at[
        i - 1, "Adj Close"]) * 100
    df_vix.at[i, "VIX_change_in_day"] = ((df_vix.at[i - 1, "Close"] - df_vix.at[i - 1, "Open"]) / df_vix.at[
        i - 1, "Adj Close"]) * 100
    df_vix.at[i, "VIX_change_in_day_open_high"] = ((df_vix.at[i - 1, "Open"] - df_vix.at[i - 1, "High"]) / df_vix.at[
        i - 1, "Adj Close"]) * 100
    df_vix.at[i, "VIX_change_in_day_open_low"] = ((df_vix.at[i - 1, "Open"] - df_vix.at[i - 1, "Low"]) / df_vix.at[
        i - 1, "Adj Close"]) * 100
    df_vix.at[i, "VIX_change_in_day_high_low"] = ((df_vix.at[i - 1, "High"] - df_vix.at[i - 1, "Low"]) / df_vix.at[
        i - 1, "Adj Close"]) * 100

df_vix

df_vix.rename(columns={'Open': 'Open_VIX', 'High': 'High_VIX', 'Low': 'Low_VIX', 'Close': 'Close_VIX',
                       'Adj Close': 'Adj Close_VIX', 'Volume': 'Volume_VIX'}, inplace=True)

df_vix

df
df

# df_apple.rename(columns = {'Open':'Open_appl','High':'High_appl','Low':'Low_appl','Close':'Close_appl','Adj Close':'Adj Close_appl','Volume':'Volume_appl'}, inplace = True)

# df_apple


df = pd.merge(df, df_snopi, on='Date')

df

df = pd.merge(df, df_vix, on='Date')

df

df.drop("Volume_VIX", axis='columns', inplace=True)

df

df = df.sort_values(["company_name", "Date"], ascending=[True, False]).reset_index(drop=False)
df.drop(["index"], axis=1, inplace=True)

df

for i in df.index[1:]:
    if df.at[i, "company_name"] == df.at[i - 1, "company_name"]:
        df.at[i, "stock_change_target"] = ((df.at[i - 1, "Adj Close"] - df.at[i, "Adj Close"]) / df.at[
            i - 1, "Adj Close"]) * 100
        df.at[i, "stock_change_in_day_Adj"] = ((df.at[i - 1, "Adj Close"] - df.at[i - 1, "Open"]) / df.at[
            i - 1, "Adj Close"]) * 100
        df.at[i, "stock_change_in_day"] = ((df.at[i - 1, "Close"] - df.at[i - 1, "Open"]) / df.at[
            i - 1, "Adj Close"]) * 100
        df.at[i, "stock_change_in_day_open_high"] = ((df.at[i - 1, "Open"] - df.at[i - 1, "High"]) / df.at[
            i - 1, "Adj Close"]) * 100
        df.at[i, "stock_change_in_day_open_low"] = ((df.at[i - 1, "Open"] - df.at[i - 1, "Low"]) / df.at[
            i - 1, "Adj Close"]) * 100
        df.at[i, "stock_change_in_day_high_low"] = ((df.at[i - 1, "High"] - df.at[i - 1, "Low"]) / df.at[
            i - 1, "Adj Close"]) * 100

df.info

df.info()

grouped = df.groupby('company_name')


# Define the min-max normalization function
def min_max_normalize(x):
    return (x - x.min()) / (x.max() - x.min())


# Apply the min-max normalization function to the Volume column for each group
df['Volume_normalized'] = grouped['Volume'].transform(min_max_normalize)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df['Volume_GSPC_normalized'] = scaler.fit_transform(df[['Volume_GSPC']])

df

df = df.sort_values(["company_name", "Date"], ascending=[True, True]).reset_index(drop=False)

df.drop(["index"], axis=1, inplace=True)
df.info()
# df.to_csv('out.csv')
ma_day = [10, 20, 50]

for ma in ma_day:
    column_name = f"stock_MA_for_{ma}_days"
    df[column_name] = df['stock_change_target'].rolling(ma).mean()

ma_day = [10, 20, 50]

for ma in ma_day:
    column_name = f"GSPC_MA_for_{ma}_days"
    df[column_name] = df['GSPC_change_in_day_Adj'].rolling(ma).mean()

ma_day = [10, 20, 50]

for i in ma_day:
    column_name = f"VIX_MA_for_{ma}_days"
    df[column_name] = df['VIX_change_in_day_Adj'].rolling(ma).mean()

df = df.sort_values(["company_name", "Date"], ascending=[True, False]).reset_index(drop=False)

df.drop(["index"], axis=1, inplace=True)

day_names = ['is_Monday', 'is_Tuesday', 'is_Wednesday', 'is_Thursday', 'is_Friday', 'is_Saturday', 'is_Sunday']

for i, x in enumerate(day_names):
    df[x] = df['Date'].dt.weekday.apply(lambda x: 1 if x == i else 0)

df

for i, row in df.iterrows():
    # if df.at[i, "stock_change_target"] >= 0.7:
    #     df.at[i, "target"] = 1
    # else :
    #     if df.at[i, "stock_change_target"] < -0.7:
    #         df.at[i, "target"] = -1
    #     else:
    #         df.at[i, "target"] = 0
    if df.at[i, "stock_change_target"] >= 0:
        df.at[i, "target"] = 1
    else :
            df.at[i, "target"] = -1




df.info

dummy_df = pd.get_dummies(df["company_name"], prefix="company")

# concatenate the dummy variables with the original dataframe
df = pd.concat([df, dummy_df], axis=1)

# drop the original "company_name" column
# df.drop("company_name", axis=1, inplace=True)

print(df.head())

df.info

df = df.dropna(subset=df.columns[1:], axis=0)

df.info

df

df_data_for_models = df.drop(
    columns=['Volume', 'Open', 'Open_GSPC', 'High_GSPC', 'Low_GSPC', 'High', 'Low', 'Close', 'Adj Close', 'Open',
             'High', 'Low', 'Close_GSPC', 'Adj Close_GSPC', 'Open_VIX', 'High_VIX', 'Low_VIX', 'Close_VIX',
             'Adj Close_VIX', ])

df_data_for_models

df_data_for_models.info()

df_data_for_models = df_data_for_models.dropna()

X = df_data_for_models.drop('target', axis=1)
y = df_data_for_models.target

X

y

X_scaled = df_data_for_models.drop(columns=['Date', "company_name", 'target', 'stock_change_target'])
y = df_data_for_models['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=1234)
train_df = pd.merge(left=X_train, right=y_train, left_index=True, right_index=True)
test_df = pd.merge(left=X_test, right=y_test, left_index=True, right_index=True)
train_df.head(10)

from sklearn import preprocessing

encoder = preprocessing.LabelEncoder()

# encoding train labels
encoder.fit(y_train)
y_train_n = encoder.transform(y_train)

# encoding test labels
encoder.fit(y_test)
y_test_n = encoder.transform(y_test)
print(y_test[0:5].to_list())
print(y_test_n[0:5])

X_train.info()

X_test.info()

X = df_data_for_models.drop(columns=['Date', "company_name", 'target', 'stock_change_target'])
y = df_data_for_models['target']

X

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
train_df = pd.merge(left=X_train, right=y_train, left_index=True, right_index=True)
test_df = pd.merge(left=X_test, right=y_test, left_index=True, right_index=True)
train_df.head(5)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=30, criterion='entropy')

rfc.fit(X_train, y_train)

y_predict = rfc.predict(X_test)

print(confusion_matrix(y_test, y_predict))
print('----------------------------------------------------------')
print(classification_report(y_test, y_predict))


from sklearn.model_selection import cross_val_score
auc_scores = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc')
auc_scores



accuracy_scores = cross_val_score(rfc, X, y, cv=10, scoring='accuracy')
accuracy_scores

print('Mean values')
print('auc:', auc_scores.mean())
print('accuracy: ', accuracy_scores.mean())

train_df['predicted_y'] = rfc.predict(X_train)
test_df['predicted_y'] = rfc.predict(X_test)

# Create new dataframes with train and test data
df_rf_s_train = pd.concat(
    [df_data_for_models, pd.DataFrame(rfc.predict(X_train), index=X_train.index, columns=['predicted_y'])], axis=1)
df_rf_s_test = pd.concat([test_df, pd.DataFrame(rfc.predict(X_test), index=X_test.index, columns=['predicted_y'])],
                         axis=1)

# make predictions on training and test data
train_preds = rfc.predict(X_train)
test_preds = rfc.predict(X_test)

create a column for predicted values in original dataframe
df_data_for_models['predicted_y'] = np.concatenate((train_preds, test_preds), axis=0)

import pandas as pd

# create the id_name table
id_name = pd.DataFrame({
    'id': [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
           32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
    'company_name': ['Apple Inc.', 'Meta', 'Microsoft', 'Google', 'Nike', 'Netflix', 'Adobe', 'Zoom Inc.', 'Nvidia',
                     'HubSpot', 'Laboratory Corp', 'Energy Transfer', 'KBR Inc', 'Walmart Inc.', 'Stryker Corp',
                     'Deere & Co', 'Apellis', 'Home Depot Inc.', 'Jabil Inc.', 'Pure Storage', 'Palo Alto',
                     'Block Inc.', 'PPL Corp', 'Arch Capital', 'Datadog Inc.', 'Alphabet Inc.', 'Allianz SE',
                     'Papa John\'s', 'Stem Inc.', 'General Electric', 'General Motors', 'SiTime Corp.', 'Box Inc.',
                     'Seagen Inc.', 'Corteva Inc.', 'Dynatrace Inc.', 'Planet Fitness', 'Chart Industries',
                     'Privia Health', 'Vinci S.A', 'Pinterest', 'Cenovus Energy', 'Vulcan Materials', 'Globant SA',
                     'Copa Holdings', 'TechnipFMC', 'Agnico Eagle', 'Waste Conn.']
})

# merge the two tables on the "name" column
df_data_for_models = pd.merge(df_data_for_models, id_name, on='company_name')

# print the updated table
print(df_data_for_models)

df_data_for_models

df_data_for_models.info()

df_data_for_models_n = df_data_for_models[df_data_for_models['Date'] >= pd.to_datetime('2023-02-01')]

df_data_for_models_n


#
# import mysql.connector
#
# mydb = mysql.connector.connect(
#     host='45.136.70.170',
#     user='stockmaven_dev',
#     password='fnSvAh79baKF'
# )
#
# print(mydb)
#
# mycursor = mydb.cursor()
#
# mycursor.execute("SHOW DATABASES")
#
# for x in mycursor:
#     print(x)
#
# import mysql.connector
#
# mydb = mysql.connector.connect(
#     host='45.136.70.170',
#     user='stockmaven_dev',
#     password='fnSvAh79baKF',
#     database="stockmaven_dev"
# )
#
# mycursor = mydb.cursor()
#
# mycursor.execute("SHOW TABLES")
#
# for x in mycursor:
#     print(x)
#
# import mysql.connector
# from mysql.connector import Error
#
# # Connect to the MySQL database
# try:
#     mydb = mysql.connector.connect(
#         host='45.136.70.170',
#         user='stockmaven_dev',
#         password='fnSvAh79baKF',
#         database="stockmaven_dev"
#     )
#
#     mycursor = mydb.cursor()
#
#     # Iterate over each row in the DataFrame
#     for index, row in df_data_for_models_n.iterrows():
#         try:
#             # Extract the necessary values from the DataFrame
#             prediction = row['predicted_y']
#             stockid = row['id']
#             date = row['Date'].strftime('%Y-%m-%d %H:%M:%S')
#
#             # Prepare the INSERT query
#             sql = "INSERT INTO stock_prediction (prediction, stockid, date) VALUES (%s, %s, %s)"
#             values = (prediction, stockid, date)
#
#             # Execute the INSERT query
#             mycursor.execute(sql, values)
#             mydb.commit()
#             print("Data inserted successfully!")
#         except Error as e:
#             print("Error inserting row:", e)
#             continue
#
#         # Commit the changes to the database
#         # mydb.commit()
#         # print("Data inserted successfully!")
#
# except Error as e:
#     print("Error connecting to MySQL database:", e)
#
# finally:
#     # Close the database connection
#     if mydb.is_connected():
#         mycursor.close()
#         mydb.close()