import pandas as pd
import json

# 1. Load your raw data (using your existing logic)
cols = ['units', 'cycles'] + [f"op_setting_{i}" for i in range(1,4)] + [f"sensor_{i}" for i in range(1,22)]
df = pd.read_csv('train_FD001.txt', sep=r"\s+", header=None, names=cols)

# 2. Pick one engine and take the first 30 rows
engine_id = 1
# Drop metadata to keep only the 21 sensors
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
window_data = df[df['units'] == engine_id].iloc[0:30][sensor_cols]

# 3. Create the JSON structure
payload = {
    "engine_id": int(engine_id),
    "window": window_data.values.tolist()
}

print(json.dumps(payload))