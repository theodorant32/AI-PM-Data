import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

n_days = 1000
start_date = datetime(2021, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(n_days)]

base_temp = 15
seasonal = 10 * np.sin(2 * np.pi * np.arange(n_days) / 365)
trend = 0.01 * np.arange(n_days)
noise = np.random.normal(0, 3, n_days)
temperature = base_temp + seasonal + trend + noise

humidity = 60 + 15 * np.sin(2 * np.pi * np.arange(n_days) / 365 + 0.5) + np.random.normal(0, 10, n_days)
humidity = np.clip(humidity, 20, 100)

precipitation = np.maximum(0, np.random.exponential(2, n_days) * (1 + 0.5 * np.sin(2 * np.pi * np.arange(n_days) / 365)))

df = pd.DataFrame({
    'lastupdated': dates,
    'temperature': temperature,
    'humidity': humidity,
    'precipitation': precipitation
})

df.loc[np.random.choice(n_days, size=50), 'temperature'] = np.nan
df.loc[np.random.choice(n_days, size=30), 'humidity'] = np.nan

df.to_csv('data/GlobalWeatherRepository.csv', index=False)
print(f'Generated {len(df)} rows of sample weather data')
