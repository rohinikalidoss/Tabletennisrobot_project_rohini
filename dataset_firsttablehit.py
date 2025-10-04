import h5py
import numpy as np
import matplotlib.pyplot as plt

def extract_first_table_hits(file_path, table_height_min=0.76, table_height_max=0.76):
    table_length, table_width = 2.74, 1.525
    hit_x, hit_y, hit_z = [], [], []

    with h5py.File(file_path, "r") as f:
        traj_group = f["originals"]

        for key in traj_group.keys():
            positions = traj_group[key]["positions"][:]
            if positions.shape[0] < 2:
                continue

            for i in range(len(positions)):
                x, y, z = positions[i]
                # Check if z is within allowed range
                if table_height_min <= z <= table_height_max:
                    if 0 <= x <= table_length and -table_width/2 <= y <= table_width/2:
                        hit_x.append(x)
                        hit_y.append(y)
                        hit_z.append(z)
                        break  # only first hit
                        
    return np.array(hit_x), np.array(hit_y), np.array(hit_z)



def visualize_first_table_hits(file_path):
    hit_x, hit_y, hit_z = extract_first_table_hits(file_path)

    if len(hit_x) == 0:
        print("No table hits found.")
        return

    print(f"✅ Collected {len(hit_x)} first table hits")

    # 2D Top View Plot (X-Y)
    plt.figure(figsize=(8, 6))
    plt.scatter(hit_x, hit_y, c="red", s=15, alpha=0.6, label="First Table Hit")

    # Draw table outline
    table_length, table_width = 2.74, 1.525
    plt.gca().add_patch(plt.Rectangle((0, -table_width/2), table_length, table_width,
                                      edgecolor="black", facecolor="lightgray", alpha=0.3))
    plt.axvline(2.74/2, color="white", lw=2)  # net line

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("First Table Hits (Top View)")
    plt.legend()
    plt.axis("equal")
    plt.show()


# Example usage
file_path = "sweep_dataset/MN5008_grid_data_all.hdf5"
visualize_first_table_hits(file_path)
