import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Create directory for visualizations
os.makedirs("CBH_Shift_Offers/reports/figures", exist_ok=True)

# Load processed data
pickle_filename = 'CBH_Shift_Offers/data/interim/01_Processed_Data.pkl'
df = pd.read_pickle(pickle_filename)

# Step 7: Data Analysis Report Generation
# =============================================================================
# Create time period columns for analysis
df['YEAR'] = df['SHIFT_START_AT'].dt.year.astype(int)
df['MONTH'] = df['SHIFT_START_AT'].dt.month.astype(int)
df['WEEK'] = df['SHIFT_START_AT'].dt.isocalendar().week.astype(int)
df['MONTH_YEAR'] = df['SHIFT_START_AT'].dt.to_period('M')
df['DATE'] = df['SHIFT_START_AT'].dt.date

# 1. Highest total charge by location
def top_workplaces_by_charge(period):
    result = df.groupby(['WORKPLACE_ID', period])['TOTAL_CHARGE_RATE'].sum().reset_index()
    # Convert period to string for better plotting
    result[period] = result[period].astype(str)
    return result.sort_values('TOTAL_CHARGE_RATE', ascending=False).head(10)

weekly_charge = top_workplaces_by_charge('WEEK')
monthly_charge = top_workplaces_by_charge('MONTH')
yearly_charge = top_workplaces_by_charge('YEAR')

# 2. Most common values for viewed offers
viewed_df = df[df['OFFER_VIEWED_AT'].notnull()]

duration_counts = viewed_df['DURATION'].value_counts().head(10).reset_index()
duration_counts.columns = ['DURATION', 'VIEW_COUNT']

pay_rate_counts = viewed_df['PAY_RATE'].value_counts().head(10).reset_index()
pay_rate_counts.columns = ['PAY_RATE', 'VIEW_COUNT']

charge_rate_counts = viewed_df['CHARGE_RATE'].value_counts().head(10).reset_index()
charge_rate_counts.columns = ['CHARGE_RATE', 'VIEW_COUNT']

# 3. Workplaces with most verified and NCNS
def count_events(event_col, period):
    event_df = df[df[event_col]]
    result = event_df.groupby(['WORKPLACE_ID', period]).size().reset_index()
    result.columns = ['WORKPLACE_ID', period, 'COUNT']
    # Convert period to string for better plotting
    result[period] = result[period].astype(str)
    return result.sort_values('COUNT', ascending=False).head(10)

weekly_verified = count_events('IS_VERIFIED', 'WEEK')
monthly_verified = count_events('IS_VERIFIED', 'MONTH')
yearly_verified = count_events('IS_VERIFIED', 'YEAR')

weekly_ncns = count_events('IS_NCNS', 'WEEK')
monthly_ncns = count_events('IS_NCNS', 'MONTH')
yearly_ncns = count_events('IS_NCNS', 'YEAR')

# Visualization Section
# =============================================================================
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 10, 'figure.figsize': (12, 6)})

# 1. Revenue Analysis Visualizations
# Top locations by charge (Bar Charts)
def plot_top_charge(data, period, title):
    plt.figure(figsize=(12, 7))
    # Use period directly as hue since we converted to string
    sns.barplot(data=data, x='WORKPLACE_ID', y='TOTAL_CHARGE_RATE', hue=period)
    plt.title(f'Top Locations by Total Charge ({title})')
    plt.xlabel('Workplace ID')
    plt.ylabel('Total Charge ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'CBH_Shift_Offers/reports/figures/top_charge_{title.lower()}.png')
    plt.close()

plot_top_charge(weekly_charge, 'WEEK', 'Weekly')
plot_top_charge(monthly_charge, 'MONTH', 'Monthly')
plot_top_charge(yearly_charge.head(5), 'YEAR', 'Yearly')

# Revenue Trends Over Time (Line Chart)
monthly_revenue = df.groupby('MONTH_YEAR')['TOTAL_CHARGE_RATE'].sum().reset_index()
monthly_revenue['MONTH_YEAR'] = monthly_revenue['MONTH_YEAR'].astype(str)

plt.figure(figsize=(14, 7))
sns.lineplot(data=monthly_revenue, x='MONTH_YEAR', y='TOTAL_CHARGE_RATE', marker='o')
plt.title('Monthly Revenue Trend')
plt.xlabel('Month')
plt.ylabel('Total Revenue ($)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('CBH_Shift_Offers/reports/figures/monthly_revenue_trend.png')
plt.close()

# 2. Shift Characteristics Visualizations
# Most common durations (Pie Chart)
plt.figure(figsize=(10, 8))
duration_counts.set_index('DURATION')['VIEW_COUNT'].plot.pie(autopct='%1.1f%%')
plt.title('Distribution of Shift Durations (Viewed Offers)')
plt.ylabel('')
plt.tight_layout()
plt.savefig('CBH_Shift_Offers/reports/figures/duration_distribution.png')
plt.close()

# Pay vs Charge Rate Distribution (Dual Histogram)
plt.figure(figsize=(12, 7))
sns.histplot(data=viewed_df, x='PAY_RATE', bins=20, color='blue', alpha=0.5, label='Pay Rate')
sns.histplot(data=viewed_df, x='CHARGE_RATE', bins=20, color='red', alpha=0.5, label='Charge Rate')
plt.title('Distribution of Pay Rates vs Charge Rates')
plt.xlabel('Rate ($)')
plt.legend()
plt.tight_layout()
plt.savefig('CBH_Shift_Offers/reports/figures/rate_distribution.png')
plt.close()

# 3. Operational Metrics Visualizations
# Verified vs NCNS Comparison (Bar Chart)
verified_counts = df['IS_VERIFIED'].value_counts()
ncns_counts = df['IS_NCNS'].value_counts()

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
verified_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Verified Shifts')
plt.xticks([0, 1], ['Not Verified', 'Verified'], rotation=0)

plt.subplot(1, 2, 2)
ncns_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('NCNS Incidents')
plt.xticks([0, 1], ['No NCNS', 'NCNS'], rotation=0)

plt.tight_layout()
plt.savefig('CBH_Shift_Offers/reports/figures/verification_ncns_comparison.png')
plt.close()

# NCNS Trend Over Time (Line Chart)
ncns_trend = df[df['IS_NCNS']].groupby('DATE').size().reset_index(name='COUNT')
ncns_trend['DATE'] = pd.to_datetime(ncns_trend['DATE'])

plt.figure(figsize=(14, 7))
sns.lineplot(data=ncns_trend, x='DATE', y='COUNT')
plt.title('NCNS Incidents Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Incidents')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('CBH_Shift_Offers/reports/figures/ncns_trend.png')
plt.close()

# 4. Rate Analysis Visualizations
# Pay vs Charge Rate Relationship (Scatter Plot)
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='PAY_RATE', y='CHARGE_RATE', hue='DURATION', size='DURATION', sizes=(20, 200))
plt.title('Pay Rate vs Charge Rate Relationship')
plt.xlabel('Pay Rate ($)')
plt.ylabel('Charge Rate ($)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('CBH_Shift_Offers/reports/figures/pay_charge_relationship.png')
plt.close()

# Rate Distribution by Slot (Box Plot)
plt.figure(figsize=(12, 7))
sns.boxplot(data=df, x='SLOT', y='PAY_RATE')
plt.title('Pay Rate Distribution by Time Slot')
plt.xlabel('Time Slot')
plt.ylabel('Pay Rate ($)')
plt.tight_layout()
plt.savefig('CBH_Shift_Offers/reports/figures/payrate_by_slot.png')
plt.close()

# Generate Report
print("\n" + "="*80)
print("DATA ANALYSIS REPORT")
print("="*80)

# Section 1: Highest Total Charge
print("\n1. LOCATIONS WITH HIGHEST TOTAL CHARGE")
print("-"*80)
print("\nWeekly Top Workplaces:")
print(weekly_charge.to_string(index=False))
print("\nMonthly Top Workplaces:")
print(monthly_charge.to_string(index=False))
print("\nYearly Top Workplaces:")
print(yearly_charge.to_string(index=False))

# Section 2: Most Common Values
print("\n\n2. MOST POPULAR SHIFT CHARACTERISTICS FOR VIEWED OFFERS")
print("-"*80)
print("\nTop Durations:")
print(duration_counts.to_string(index=False))
print("\nTop Pay Rates:")
print(pay_rate_counts.to_string(index=False))
print("\nTop Charge Rates:")
print(charge_rate_counts.to_string(index=False))

# Section 3: Verified and NCNS Events
print("\n\n3. WORKPLACES WITH MOST SHIFT VERIFICATIONS")
print("-"*80)
print("\nWeekly Verified Shifts:")
print(weekly_verified.to_string(index=False))
print("\nMonthly Verified Shifts:")
print(monthly_verified.to_string(index=False))
print("\nYearly Verified Shifts:")
print(yearly_verified.to_string(index=False))

print("\n\nWORKPLACES WITH MOST NO-CALL NO-SHOW (NCNS) INCIDENTS")
print("-"*80)
print("\nWeekly NCNS Incidents:")
print(weekly_ncns.to_string(index=False))
print("\nMonthly NCNS Incidents:")
print(monthly_ncns.to_string(index=False))
print("\nYearly NCNS Incidents:")
print(yearly_ncns.to_string(index=False))
print("="*80)

# Step 8: Verification of processed data
print("\n\nPROCESSED DATA VERIFICATION")
print("-"*80)
print("First 5 rows of the processed data:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nSummary statistics:")
print(df.describe(include='all'))