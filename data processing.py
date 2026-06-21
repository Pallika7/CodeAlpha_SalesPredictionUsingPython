import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# DATA INSPECTION & CLEANING
# Load the raw advertising dataset
try:
    df = pd.read_csv('Advertising data.csv')
    print("Successfully loaded 'Advertising data.csv'")
except FileNotFoundError:
    print("Error: 'Advertising data.csv' not found. Please check the file path.")
    exit()

#Rename the vague index column to 'Sr. No.'
if 'Unnamed: 0' in df.columns:
    df.rename(columns={'Unnamed: 0': 'Sr. No.'}, inplace=True)
    print("-> Renamed 'Unnamed: 0' column to 'Sr. No.'")

# Check and print data integrity stats
missing_count = df.isnull().sum().sum()
duplicate_count = df.duplicated().sum()

print(f"-> Total missing values found: {missing_count}")
print(f"-> Total duplicate rows found: {duplicate_count}")

#Final cleaning drop step (safeguard)
df = df.drop_duplicates().dropna()

# ADVANCED FEATURE ENGINEERING & SELECTION

# Overall Marketing Investment Size
df['Total_Spend'] = df['TV'] + df['Radio'] + df['Newspaper']

# Strategic Budget Share Allocations (added 1e-5 to prevent division-by-zero errors)
df['TV_Share'] = df['TV'] / (df['Total_Spend'] + 1e-5)
df['Radio_Share'] = df['Radio'] / (df['Total_Spend'] + 1e-5)
df['Newspaper_Share'] = df['Newspaper'] / (df['Total_Spend'] + 1e-5)

# Cross-Channel Interaction Effect (Synergy / Multiplier Effect)
df['TV_Radio_Interaction'] = df['TV'] * df['Radio']

#Log Transformations to account for Diminishing Returns (Market Saturation)
df['log_TV'] = np.log1p(df['TV'])
df['log_Radio'] = np.log1p(df['Radio'])

print("Successfully engineered 7 new feature columns")
# DATA VALIDATION & EXPORT
#Show descriptive preview of the new dataset structure
print("\n NEW ENHANCED DATASET PREVIEW (HEAD)")
print(df.head())

#Rank features by their statistical correlation to Sales
print("\n CORRELATION MATRIX RANKING WITH TARGET (SALES)")
target_correlations = df.corr()['Sales'].sort_values(ascending=False)
print(target_correlations)

#Save the final enriched dataframe to a new file
output_filename = 'engineered_advertising_data.csv'
df.to_csv(output_filename, index=False)
print(f"\nCleaned and engineered dataset saved as '{output_filename}'")

#Generate visual validation plot
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap with Engineered Features')
plt.tight_layout()
plt.savefig('engineered_features_correlation.png')
print("Visual verification plot generated and saved as 'engineered_features_correlation.png'")