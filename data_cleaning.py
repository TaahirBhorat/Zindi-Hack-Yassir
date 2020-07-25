import pandas as pd
df = pd.read_csv('Train.csv')
weather_df = pd.read_csv('Weather.csv')

df['Timestamp'] = pd.to_datetime(df['Timestamp'], infer_datetime_format=True)
weather_df['date'] = pd.to_datetime(weather_df['date'], infer_datetime_format=True)
