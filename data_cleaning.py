import pandas as pd
trips_df = pd.read_csv('Train.csv')
weather_df = pd.read_csv('Weather.csv')

trips_df['Timestamp'] = pd.to_datetime(trips_df['Timestamp'], infer_datetime_format=True)
weather_df['date'] = pd.to_datetime(weather_df['date'], infer_datetime_format=True)

trips_df['date'] = trips_df['Timestamp'].dt.date
weather_df['date'] = weather_df['date'].dt.date

df = pd.merge(trips_df, weather_df, how='left', on='date').drop('date', axis=1)
del weather_df, trips_df

# remove trips with average speed over 200
df = df[(df['Trip_distance']/1000)/(df['ETA']/(60*60)) <= 200]
