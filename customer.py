# customer.py
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Helper: print header
def header(msg):
    print("\n" + "="*6 + " " + msg + " " + "="*6)

# ---------------------------
# Load dataset safely
# ---------------------------
CSV_NAME = "customer.csv"   # change if your file has a different name

if not os.path.exists(CSV_NAME):
    print(f"ERROR: '{CSV_NAME}' not found in {os.getcwd()}")
    print("Put the CSV in the same folder as this script or provide a full path.")
    sys.exit(1)

try:
    data = pd.read_csv(CSV_NAME)
except Exception as e:
    print("ERROR reading CSV:", e)
    sys.exit(1)

header("Preview of loaded data")
print(data.head())

# ---------------------------
# Basic column checks
# ---------------------------
required_cols = {'Age', 'Purchase Amount (USD)', 'Previous Purchases', 'Gender'}
missing = required_cols - set(data.columns)
if missing:
    print(f"ERROR: The following required columns are missing from the CSV: {missing}")
    print("Columns found:", list(data.columns))
    sys.exit(1)

# ---------------------------
# Data Preprocessing
# ---------------------------
data.ffill(inplace=True)   # forward fill any missing values

# ---------------------------
# Feature Engineering
# ---------------------------
# If you want cumulative total per customer you normally need a transaction-level dataset
# Here we follow original approach: cumulative across the file (simple demo)
data['Total_Purchase_Amount'] = data['Purchase Amount (USD)'].cumsum()

numeric_cols = ['Age', 'Total_Purchase_Amount', 'Previous Purchases']

# Ensure numeric types
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data.dropna(subset=numeric_cols, inplace=True)

# ---------------------------
# Feature Scaling
# ---------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[numeric_cols])

# ---------------------------
# Elbow Method to select K
# ---------------------------
header("Computing WCSS for Elbow Method")
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    km.fit(scaled_data)
    wcss.append(km.inertia_)
print("WCSS:", wcss)

plt.figure(figsize=(7,5))
plt.plot(range(1,11), wcss, marker='o', linestyle='--')
plt.title("Elbow Method (WCSS vs number of clusters)")
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show(block=True)

# ---------------------------
# Fit final KMeans (choose k=3 or pick from elbow)
# ---------------------------
k = 3
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
kmeans.fit(scaled_data)
data['Cluster'] = kmeans.labels_

header("Cluster counts")
print(data['Cluster'].value_counts())

# ---------------------------
# Visualization: clusters
# ---------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(data=data, x='Age', y='Total_Purchase_Amount', hue='Cluster', palette='viridis', s=80)
plt.title("Customer Segmentation (Age vs Total Purchase)")
plt.xlabel("Age")
plt.ylabel("Total Purchase Amount")
plt.grid(True)
plt.show(block=True)

# ---------------------------
# Analysis: cluster means
# ---------------------------
header("Cluster means")
cluster_means = data.groupby('Cluster')[numeric_cols].mean()
print(cluster_means)

# ---------------------------
# Hist & Boxplot
# ---------------------------
plt.figure(figsize=(7,5))
plt.hist(data['Purchase Amount (USD)'].dropna(), bins=15)
plt.title("Purchase Amount Distribution")
plt.xlabel("Purchase Amount (USD)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show(block=True)

plt.figure(figsize=(7,5))
sns.boxplot(data=data, x='Gender', y='Purchase Amount (USD)')
plt.title("Purchase Amount by Gender")
plt.grid(True)
plt.show(block=True)

# ---------------------------
# AI Prediction: example new customer
# ---------------------------
new_customer = pd.DataFrame({
    'Age': [32],
    'Total_Purchase_Amount': [2500],
    'Previous Purchases': [8]
})

# scale with same scaler and predict
try:
    new_scaled = scaler.transform(new_customer[numeric_cols])
    pred = kmeans.predict(new_scaled)
    print(f"\nAI Prediction: new customer assigned to cluster {pred[0]}")
except Exception as e:
    print("Prediction error:", e)

header("Script finished")
