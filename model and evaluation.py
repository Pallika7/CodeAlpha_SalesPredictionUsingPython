import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data and recreate basic interaction
df = pd.read_csv('Advertising data.csv')
if 'Unnamed: 0' in df.columns:
    df.rename(columns={'Unnamed: 0': 'Sr. No.'}, inplace=True)

df['TV_Radio_Interaction'] = df['TV'] * df['Radio']

# Define inputs and targets
X_baseline = df[['TV', 'Radio', 'Newspaper']]
X_enhanced = df[['TV', 'Radio', 'Newspaper', 'TV_Radio_Interaction']]
y = df['Sales']

# Split
X_train_base, X_test_base, y_train, y_test = train_test_split(X_baseline, y, test_size=0.2, random_state=42)
X_train_enh, X_test_enh, _, _ = train_test_split(X_enhanced, y, test_size=0.2, random_state=42)

# Fit Models
model_base = LinearRegression()
model_base.fit(X_train_base, y_train)

model_enh = LinearRegression()
model_enh.fit(X_train_enh, y_train)

# Predict
y_pred_base = model_base.predict(X_test_base)
y_pred_enh = model_enh.predict(X_test_enh)

# Print results
print(" MODEL EVALUATION ")
print("Model 1: Baseline (Raw Features Only)")
print("R2 Score: ", round(r2_score(y_test, y_pred_base), 4))
print("MAE:      ", round(mean_absolute_error(y_test, y_pred_base), 4))
print("RMSE:     ", round(np.sqrt(mean_squared_error(y_test, y_pred_base)), 4))

print("\nModel 2: Synergy-Enhanced (With TV*Radio Interaction)")
print("R2 Score: ", round(r2_score(y_test, y_pred_enh), 4))
print("MAE:      ", round(mean_absolute_error(y_test, y_pred_enh), 4))
print("RMSE:     ", round(np.sqrt(mean_squared_error(y_test, y_pred_enh)), 4))

print("\n MODEL PARAMETERS")
print("Model 2 Intercept:", round(model_enh.intercept_, 4))
features = ['TV', 'Radio', 'Newspaper', 'TV_Radio_Interaction']
for feat, coef in zip(features, model_enh.coef_):
    print(f"Coefficient for {feat}: {coef:.6f}")