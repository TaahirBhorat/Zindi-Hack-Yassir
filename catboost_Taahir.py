import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import os
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from math import sqrt

IN_COLLAB = False

if IN_COLLAB:
    files_directory = '/content/drive/My Drive/Zindi/'
else:
    files_directory = ''


def pre_process(df):
    # cluster lat long
    arr = np.vstack((df[['Origin_lat', 'Origin_lon']].values,
                        df[['Destination_lat', 'Destination_lon']].values))
    sample_ind = np.random.permutation(len(arr))
    kmeans =MiniBatchKMeans(n_clusters=90, batch_size=10000).fit(arr[sample_ind])
    
    df.loc[:, 'pickup_cluster'] = kmeans.predict(df[['Origin_lat', 'Origin_lon']])
    df.loc[:, 'dropoff_cluster'] = kmeans.predict(df[['Destination_lat', 'Destination_lon']])
    
    arr = np.vstack((df[['Origin_lat', 'Origin_lon']].values,
                        df[['Destination_lat', 'Destination_lon']].values))
    
    # PCA lat long
    pca = PCA().fit(arr)
    df['pickup_pca0'] = pca.transform(df[['Origin_lat', 'Origin_lon']])[:, 0]
    df['pickup_pca1'] = pca.transform(df[['Origin_lat', 'Origin_lon']])[:, 1]
    df['dropoff_pca0'] = pca.transform(df[['Destination_lat', 'Destination_lon']])[:, 0]
    df['dropoff_pca1'] = pca.transform(df[['Destination_lat', 'Destination_lon']])[:, 1]
    
    StartTime = pd.to_datetime(df['Timestamp'], infer_datetime_format=True)
    
    df['is_peak_traffic'] = [1 if (5<i<9 or 15<i<20) else 0 for i in StartTime.dt.hour]
    df['Day_in_week'] = StartTime.dt.dayofweek
    df['Day_in_year'] = StartTime.dt.dayofyear
    df['Month'] = StartTime.dt.month
    df = df.drop('Timestamp', axis=1)
    
    return df


def add_weather(trips_df, weather_df):
    trips_df['Timestamp'] = pd.to_datetime(trips_df['Timestamp'], infer_datetime_format=True)
    weather_df['date'] = pd.to_datetime(weather_df['date'], infer_datetime_format=True)
    
    trips_df['date'] = trips_df['Timestamp'].dt.date
    weather_df['date'] = weather_df['date'].dt.date
    
    df = pd.merge(trips_df, weather_df, how='left', on='date').drop('date', axis=1)
    return df


def clean_training_set(trips_df):
    # remove trips with average speed over 200
    return trips_df[(trips_df['Trip_distance'] / 1000) / (trips_df['ETA'] / (60 * 60)) <= 200]


train = pd.read_csv(os.path.join(files_directory, 'Train.csv'))
submission_test_set = pd.read_csv(os.path.join(files_directory, 'Test.csv'))
weather = pd.read_csv(os.path.join(files_directory, 'Weather.csv'))

train = add_weather(train, weather)
train = clean_training_set(train)

submission_test_set = add_weather(submission_test_set, weather)

train = pre_process(train)
submission_test_set = pre_process(submission_test_set)

X = train.drop(['ETA', 'ID'], axis=1)
y = train['ETA']

print('splitting into test, validation and training sets')
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.01)
X_val, X_test, y_val, y_test = train_test_split(X, y, shuffle=True, test_size=0.5)

model = CatBoostRegressor(
    loss_function='RMSE',
    iterations=20000,
    learning_rate=1.0,
    task_type='GPU' if IN_COLLAB else 'CPU'
)

print('training catboost model')
model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    verbose=200)


rms = sqrt(mean_squared_error(y_test, model.predict(X_test)))
print('test score: ', rms, 'over', X_test.shape[0], 'test samples')

submission = pd.DataFrame({'ID': submission_test_set['ID'], 'ETA': model.predict(submission_test_set.drop('ID', axis=1))})
submission.to_csv('submission.csv', index=False)
