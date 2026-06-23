import pandas as pd

df = pd.read_parquet('/data/predict_0.parquet')
print(df.head(5))

#preparation