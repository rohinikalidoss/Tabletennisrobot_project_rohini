import h5py
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# === Table/net parameters ===
TABLE_HEIGHT = 0.76
TABLE_HALF_X = 1.37
TABLE_HALF_Y = 0.76
NET_X = 0.0

file_path = "sweep_dataset/grid_dataset_with300step.hdf5"
hit_positions = []

with h5py.File(file_path, "r") as f:
    for group_name in f['originals']:
        group = f['originals'][group_name]
        for ds_name in group:
            arr = np.array(group[ds_name])
            # Ensure arr is Nx3
            if arr.ndim == 1 and arr.size % 3 == 0:
                arr = arr.reshape(-1, 3)
            if arr.ndim != 2 or arr.shape[1] < 3:
                continue

            # Filter: after net crossing
            after_net = arr[arr[:, 0] > NET_X]

            # Loosen tolerance for table hits
            tolerance = 0.1
            on_table = after_net[np.abs(after_net[:, 2] - TABLE_HEIGHT) < tolerance]

            if len(on_table):
                hit_positions.append(on_table[:, :2])

if not hit_positions:
    raise ValueError("No hit positions found. Check filtering or dataset content.")

all_hits = np.vstack(hit_positions)

# === Summary statistics ===
print(f"Total hit points: {len(all_hits)}")
print(f"X range: {all_hits[:,0].min():.3f} to {all_hits[:,0].max():.3f}")
print(f"Y range: {all_hits[:,1].min():.3f} to {all_hits[:,1].max():.3f}")

# === Cluster without scaling first ===
k = 9
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(all_hits)
centroids = kmeans.cluster_centers_

# Print counts per cluster
unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Cluster {u}: {c} points")

# === Plot top-down view ===
plt.figure(figsize=(10, 6))
plt.fill_between([-TABLE_HALF_X, TABLE_HALF_X],
                 -TABLE_HALF_Y, TABLE_HALF_Y,
                 color='lightblue', alpha=0.2, label='Table')
plt.axvline(x=NET_X, color='gray', linestyle='--', linewidth=2, label='Net')

colors = plt.cm.tab10.colors
for cluster_id in np.unique(labels):
    mask = labels == cluster_id
    plt.scatter(all_hits[mask, 0], all_hits[mask, 1],
                s=20, alpha=0.7, label=f"Cluster {cluster_id}")

plt.scatter(centroids[:, 0], centroids[:, 1],
            c='black', s=100, marker='x', label='Centroids')

plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(-TABLE_HALF_X, TABLE_HALF_X)
plt.ylim(-TABLE_HALF_Y, TABLE_HALF_Y)
plt.xlabel("X position on table (m)")
plt.ylabel("Y position on table (m)")
plt.title("Ball Hit Position Clusters After Net Crossing (Summary Printed Above)")
plt.legend()
plt.show()
