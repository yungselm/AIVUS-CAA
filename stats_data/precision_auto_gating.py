import pandas as pd
import matplotlib.pyplot as plt

df_rest = pd.read_csv('output/icc_auto_manual_rest.csv')
df_stress = pd.read_csv('output/icc_auto_manual_stress.csv')

df_rest['diff'] = abs(df_rest['ivus_frame_auto'] - df_rest['frame'])
df_stress['diff'] = abs(df_stress['ivus_frame_auto'] - df_stress['frame'])
df_rest['correct'] = df_rest['diff'] <= 2.0
df_stress['correct'] = df_stress['diff'] <= 2.0

# get percent of True to length total
rest_accuracy = df_rest['correct'].mean() * 100
stress_accuracy = df_stress['correct'].mean() * 100

print(f"Rest accuracy: {rest_accuracy:.2f}%")
print(f"Stress accuracy: {stress_accuracy:.2f}%")

# number of missing peaks
missing_rest = df_rest['ivus_frame_auto'].isna().sum()
missing_stress = df_stress['ivus_frame_auto'].isna().sum()
print(f"Missing peaks (rest): {missing_rest}")
print(f"Missing peaks (stress): {missing_stress}")