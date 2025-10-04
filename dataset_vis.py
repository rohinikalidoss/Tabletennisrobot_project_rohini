import h5py
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Function to validate trajectories
# ==========================
def validate_trajectories(file_path, shot_ids):

    table_length = 2.74
    table_width = 1.525
    table_height = 0.76
    net_x = table_length / 2      
    net_height = 0.1525           

    with h5py.File(file_path, "r", libver="earliest", swmr=True) as f:

        traj_group = f["originals"]

        invalid_shots = {
            "below_ground": [],
            "hit_net": [],
            "before_net": []
        }

        for sid in shot_ids:
        
            key = f"{sid:05d}"  
            if key not in traj_group:
                continue

            try:
                positions = traj_group[key]["positions"][:]
                if positions.shape[0] == 0:
                    continue

                
                x_vals = positions[:, 0]  # X -> table length direction
                y_vals = positions[:, 1]  # Y -> table width direction
                z_vals = positions[:, 2]  # Z -> height

                if np.any(z_vals < 0):
                    invalid_shots["below_ground"].append(key)

                
                if np.max(x_vals) < net_x:
                    invalid_shots["before_net"].append(key)

                
                net_idx = np.argmin(np.abs(x_vals - net_x))
                if z_vals[net_idx] < (table_height + net_height):
                    invalid_shots["hit_net"].append(key)

            except Exception as e:
                print(f"Skipping shot {key}: {e}")

   
    print("\n=== Validation Results ===")
    print(f"Below Ground (z < 0): {invalid_shots['below_ground']}")
    print(f"Crossed Before Net (x < 1.37): {invalid_shots['before_net']}")
    print(f"Hit the Net (z < 0.9125 at net): {invalid_shots['hit_net']}")

    return invalid_shots


# ==========================
# Function to visualize trajectories
# ==========================
def plot_3d_trajectories_from_dataset(file_path, shot_ids):
    with h5py.File(file_path, "r", libver="earliest", swmr=True) as f:
        traj_group = f["originals"]  
        shot_trajectories = []
        valid_ids = []

        for sid in shot_ids:
            key = f"{sid:05d}"
            if key in traj_group:
                try:
                    positions = traj_group[key]["positions"][:]
                    if positions.shape[0] > 0:
                        shot_trajectories.append(positions)
                        valid_ids.append(key)
                except Exception as e:
                    print(f"Skipping shot {key}: {e}")

    if not shot_trajectories:
        print("No valid trajectories found.")
        return

    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.tab10(np.linspace(0, 1, len(shot_trajectories)))

    for i, trajectory in enumerate(shot_trajectories):
        ax.plot(
            trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            color=colors[i], linewidth=2.5, alpha=0.8, label=f"Shot {valid_ids[i]}"
        )

    table_length, table_width, table_height = 2.74, 1.525, 0.76
    xx, yy = np.meshgrid(np.linspace(0, table_length, 10),
                         np.linspace(-table_width/2, table_width/2, 10))
    zz = np.full_like(xx, table_height)
    ax.plot_surface(xx, yy, zz, alpha=0.4, color='lightgray', edgecolor='black', linewidth=0.5)

  
    net_x = table_length / 2
    net_height = 0.1525
    ax.plot([net_x, net_x], [-table_width/2, table_width/2],
            [table_height, table_height], 'k-', linewidth=4, label='Net Base')
    ax.plot([net_x, net_x], [-table_width/2, table_width/2],
            [table_height + net_height, table_height + net_height],
            'k-', linewidth=4, label='Net Top')
    for y in np.linspace(-table_width/2, table_width/2, 5):
        ax.plot([net_x, net_x], [y, y],
                [table_height, table_height + net_height], 'k-', linewidth=1, alpha=0.7)

    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_zlabel('Z Position (m)', fontsize=12)
    ax.set_title('3D Ball Trajectories - Table Tennis Dataset', fontsize=14)

    ax.set_xlim(-0.8, table_length + 0.5)
    ax.set_ylim(-table_width/2 - 0.2, table_width/2 + 0.2)
    ax.set_zlim(0, 1.8)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()


# ==========================
# Run Validation
# ==========================


file_path = r"sweep_dataset\MN5008_grid_data_all.hdf5"


invalid_shots = validate_trajectories(file_path, shot_ids=list(range(0, 900)))


