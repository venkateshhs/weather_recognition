import json
from datetime import datetime, timedelta, timezone

def get_timestamp(date_str: str):
    date_str_parsed, fractional_seconds_str = date_str.rsplit('_', 1)
    fractional_seconds = int(fractional_seconds_str) / 10
    dt = datetime.strptime(date_str_parsed, "%Y-%m-%d_%H-%M-%S")
    dt_with_fraction = dt + timedelta(seconds=fractional_seconds)
    dt_utc = dt_with_fraction.replace(tzinfo=timezone.utc)
    return dt_utc.strftime("%Y-%m-%d %H:%M:%S.%f")

def read_and_convert_timestamps(file_path: str):
    with open(file_path, 'r') as file:
        weather_labels = json.load(file)

    for key in weather_labels.keys():
        timestamp = get_timestamp(key)
        print(timestamp)

# Replace 'path/to/dense_weather_labels.json' with your actual file path
file_path = r'C:\Users\venkatesh\Desktop\Personal Projects\weather_recognition\resources\dense_weather_labels.json'
read_and_convert_timestamps(file_path)