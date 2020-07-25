import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from math import sqrt
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

IN_COLLAB = False
SUBMIT = True

if IN_COLLAB:
    files_directory = '/content/drive/My Drive/Zindi/'
else:
    files_directory = ''


def pre_process(df):
    # cluster lat long
    arr = np.vstack((df[['Origin_lat', 'Origin_lon']].values,
                     df[['Destination_lat', 'Destination_lon']].values))
    sample_ind = np.random.permutation(len(arr))
    kmeans = MiniBatchKMeans(n_clusters=90, batch_size=10000).fit(arr[sample_ind])
    
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
    
    ##df['is_peak_traffic'] = [1 if (5<i<9 or 15<i<20) else 0 for i in StartTime.dt.hour]
    df['Day_in_week'] = StartTime.dt.dayofweek
    df['Day_in_year'] = StartTime.dt.dayofyear
    df['Month'] = StartTime.dt.month
    df['Hour_in_Day'] = StartTime.dt.hour
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


def split_X_y(df):
    return df.drop(['ETA', 'ID'], axis=1), df['ETA']


train = pd.read_csv(os.path.join(files_directory, 'Train.csv'))
submission_test_set = pd.read_csv(os.path.join(files_directory, 'Test.csv'))
weather = pd.read_csv(os.path.join(files_directory, 'Weather.csv'))

train = add_weather(train, weather)
train = train.sort_values('Timestamp', ascending=False)
train = clean_training_set(train)

submission_test_set = add_weather(submission_test_set, weather)

train = pre_process(train)
submission_test_set = pre_process(submission_test_set)

full_train = train.copy()

print('splitting into test, validation and training sets')
test = train.iloc[:8000]
train = train.iloc[8000:]

val = train.iloc[:8000]
train = train.iloc[8000:]

X_train, y_train = split_X_y(train)
X_val, y_val = split_X_y(val)
X_test, y_test = split_X_y(test)

x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train.to_numpy().reshape(-1, 1))

params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10

if IN_COLLAB:
    params['device'] = 'gpu'

if not SUBMIT:
    print('training lgb model')
    d_train = lgb.Dataset(x_scaler.transform(X_train), label=y_scaler.transform(y_train.to_numpy().reshape(-1, 1)).reshape(-1))

    model = lgb.train(
        params,
        d_train,
        valid_sets=[lgb.Dataset(x_scaler.transform(X_val), label=y_scaler.transform(y_val.to_numpy().reshape(-1, 1)).reshape(-1))]
    )
    
    rms = sqrt(mean_squared_error(y_test, y_scaler.inverse_transform(model.predict(x_scaler.transform(X_test)))))
    print('test score: ', rms, 'over', X_test.shape[0], 'test samples')
    print('\nWARNING: NO SUBMISSION CSV WRITTEN')
else:
    d_train = lgb.Dataset(x_scaler.transform(full_train.drop(['ID', 'ETA'], axis=1)), label=y_scaler.transform(full_train['ETA'].to_numpy().reshape(-1, 1)).reshape(-1))
    
    model = lgb.train(
        params,
        d_train,
    )
    
    submission = pd.DataFrame({'ID': submission_test_set['ID'],
                               'ETA': y_scaler.inverse_transform(model.predict(x_scaler.transform(submission_test_set.drop('ID', axis=1))))})
    submission.to_csv('submission.csv', index=False)
    print('\nSubmission CSV file written')
