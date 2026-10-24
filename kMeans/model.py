# K-Means Clustering for Driver Dataset

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# 1. Load the dataset
data = pd.read_csv("driver.csv")

print("Dataset:")
print(data.head())

# 2. Select features for clustering
X = data[['mean_dist_day', 'mean_over_speed_perc']]

# 3. Create K-Means model
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

# 4. Fit the model
data['Cluster'] = kmeans.fit_predict(X)

# 5. Display cluster centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# 6. Display clustered dataset
print("\nClustered Dataset:")
print(data)

# 7. Count drivers in each cluster
print("\nNumber of drivers in each cluster:")
print(data['Cluster'].value_counts().sort_index())


# -------------------------------------------------
# 8. NEW TEST DATA
# -------------------------------------------------

# Example new driver:
# Mean distance per day = 60
# Mean over-speed percentage = 20

new_driver = pd.DataFrame({
    'mean_dist_day': [60],
    'mean_over_speed_perc': [20]
})

# Predict the cluster of new driver
new_cluster = kmeans.predict(new_driver)

print("\nNew Test Driver:")
print(new_driver)

print("\nPredicted Cluster:")
print(new_cluster[0])


# -------------------------------------------------
# 9. Distance from new driver to each cluster center
# -------------------------------------------------

distances = kmeans.transform(new_driver)

print("\nDistance from New Driver to each cluster:")
for i, distance in enumerate(distances[0]):
    print(f"Cluster {i}: {distance:.2f}")

print(
    f"\nNew driver belongs to Cluster {new_cluster[0]} "
    f"because it is closest to that cluster center."
)


# -------------------------------------------------
# 10. Plot the clusters
# -------------------------------------------------

plt.figure(figsize=(10, 6))

for cluster in range(4):

    cluster_data = data[data['Cluster'] == cluster]

    plt.scatter(
        cluster_data['mean_dist_day'],
        cluster_data['mean_over_speed_perc'],
        s=50,
        label=f'Cluster {cluster}'
    )


# Plot cluster centers
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    c='black',
    marker='X',
    label='Cluster Centers'
)


# Plot NEW TEST DRIVER
plt.scatter(
    new_driver['mean_dist_day'],
    new_driver['mean_over_speed_perc'],
    s=250,
    c='red',
    marker='*',
    label='New Test Driver'
)


# Labels
plt.xlabel("Mean Distance Per Day")
plt.ylabel("Mean Over Speed Percentage")
plt.title("K-Means Clustering of Drivers")

plt.legend()
plt.grid(True)
plt.show()












# K-Means Clustering for Driver Dataset
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Load the dataset
data = pd.read_csv("driver.csv")

# Display first 5 rows
print("Dataset:")
print(data.head())

# 2. Select the features for clustering
X = data[['mean_dist_day', 'mean_over_speed_perc']]

# 3. Create K-Means model
# We are using 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

# 4. Fit the model and assign clusters
data['Cluster'] = kmeans.fit_predict(X)

# 5. Display cluster centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# 6. Display the clustered dataset
print("\nClustered Dataset:")
print(data)

# 7. Count the number of drivers in each cluster
print("\nNumber of drivers in each cluster:")
print(data['Cluster'].value_counts().sort_index())

# 8. Plot the clusters
plt.figure(figsize=(10, 6))

for cluster in range(4):
    cluster_data = data[data['Cluster'] == cluster]

    plt.scatter(
        cluster_data['mean_dist_day'],
        cluster_data['mean_over_speed_perc'],
        s=50,
        label=f'Cluster {cluster}'
    )

# Plot cluster centers
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    c='black',
    marker='X',
    label='Cluster Centers'
)

# 9. Add labels and title
plt.xlabel("Mean Distance Per Day")
plt.ylabel("Mean Over Speed Percentage")
plt.title("K-Means Clustering of Drivers")

plt.legend()
plt.grid(True)
plt.show()
"""