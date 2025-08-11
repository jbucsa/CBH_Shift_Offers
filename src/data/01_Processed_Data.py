import pandas as pd
import numpy as np

# Step 1: Read the CSV file
df = pd.read_csv('CBH_Shift_Offers/data/raw/Problems_We_Tackle_Shift_Offers_v3_Table_12_2025-01-22T1134.csv')

# Step 2: Convert columns to specified data types
# String columns
str_cols = ['SHIFT_ID', 'WORKER_ID', 'WORKPLACE_ID', 'SLOT']
for col in str_cols:
    df[col] = df[col].astype(str)

# Datetime columns
datetime_cols = ['SHIFT_START_AT', 'SHIFT_CREATED_AT', 'OFFER_VIEWED_AT', 'CLAIMED_AT', 'CANCELED_AT', 'DELETED_AT']
for col in datetime_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Float columns
float_cols = ['PAY_RATE', 'CHARGE_RATE', 'DURATION']
for col in float_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

# Boolean columns
bool_cols = ['IS_VERIFIED', 'IS_NCNS']
for col in bool_cols:
    df[col] = df[col].astype(bool)

# Step 3: Add 'CHANGE_OF_RATE' column
df['CHANGE_OF_RATE'] = df['CHARGE_RATE'] - df['PAY_RATE']

# Step 4: Add 'TOTAL_PAY_RATE' column
df['TOTAL_PAY_RATE'] = df['PAY_RATE'] * df['DURATION']

# Step 5: Add 'TOTAL_CHARGE_RATE' column
df['TOTAL_CHARGE_RATE'] = df['CHARGE_RATE'] * df['DURATION']

# Step 6: Add 'TOTAL_CHANGE_OF_RATE' column
df['TOTAL_CHANGE_OF_RATE'] = df['TOTAL_CHARGE_RATE'] - df['TOTAL_PAY_RATE'] 

# Save as pickle file
pickle_filename = 'CBH_Shift_Offers/data/interim/01_Processed_Data.pkl'
df.to_pickle(pickle_filename)

# Step 7: View the pickle file contents
# Read the pickle file back into a DataFrame
df_view = pd.read_pickle(pickle_filename)

# Display the first 5 rows to verify
print("First 5 rows of the processed data:")
print(df_view.head())

# Optional: Display column data types
print("\nData types:")
print(df_view.dtypes)

# Optional: Display summary statistics
print("\nSummary statistics:")
print(df_view.describe(include='all'))