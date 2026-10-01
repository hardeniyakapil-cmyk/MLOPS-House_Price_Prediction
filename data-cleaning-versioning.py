import pandas as pd
import boto3
from datetime import date
import mlflow

# ==============================
# 1. Load raw CSV from Local PC
# ==============================
local_path = r"D:\Mlops_house_predication_raw_data.csv"
df = pd.read_csv(local_path)

print("============== Before Cleaning ==============")
print(df.isnull().sum())
print(f"Shape Before: {df.shape}\n")

# ==============================
# 2. Clean data
# ==============================
df_clean = df.dropna()

print("============== After Cleaning ===============")
print(df_clean.isnull().sum())
print(f"Shape After: {df_clean.shape}")

# ==============================
# 3. Save locally & Upload to S3
# ==============================
clean_path = "Mlops_house_predication_clean_v2.csv"
df_clean.to_csv(clean_path, index=False)

s3 = boto3.client("s3")
BUCKET = "ml-ops77"

def upload_processed_data(local_file):
    key = f"processed/{date.today()}/Mlops_house_predication_clean_v2.csv"
    s3.upload_file(local_file, BUCKET, key)
    print(f"\nSuccessfully uploaded cleaned data to: s3://{BUCKET}/{key}")
    return key

upload_processed_data(clean_path)

# ==============================
# 4. Features and Target
# ==============================
X = df_clean[['sqft', 'bedrooms', 'bathrooms']]
y = df_clean['price']

# ==============================
# 5. MLflow Experiment
# ==============================
mlflow.set_experiment("house-price-prediction")
print("\nExperiment configured successfully.")