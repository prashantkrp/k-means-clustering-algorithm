import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Data
data_file = r'C:\Users\Prashant kumar\PycharmProjects\pythonProject6\Mall_Customers.csv'
df = pd.read_csv(data_file)

# Print the first few rows to check the data
print(df.head())

# Step 2: Data Standardization
# Select the features for clustering
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Check if the required columns are in the DataFrame
if not all(feature in df.columns for feature in features):
    raise ValueError(f"One or more of the specified features {features} are not in the DataFrame")

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Determine the optimal number of clusters using the Elbow Method
wcss = []  # Within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Optimal number of clusters (K) can be chosen based on the elbow point in the plot
optimal_k = 3  # Example: Assume the elbow point is at K=3

# Step 4: Apply K-means Clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 5: Analysis and Visualization
# Analyze the clusters
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
print("Cluster Centers:\n", cluster_centers)

# Visualize the clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis', s=100, alpha=0.7)
plt.title('Customer Segments based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()
