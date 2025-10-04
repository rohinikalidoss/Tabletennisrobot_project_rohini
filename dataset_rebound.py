import h5py
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def filter_trajectory_before_rebound(positions, config=None):
    """
    Remove all trajectory points after first rebound (bounce).
    Based on AIMY rebound filter approach - removes second table hits.
    """
    if config is None:
        config = {
            "rebound_filter": {
                "offset": 0.0,      # Ground level offset
                "threshold": 0.1,   # Height threshold for rebound detection  
                "distance": 10      # Minimum distance between peaks
            }
        }
    
    try:
        offset = config["rebound_filter"]["offset"]
        threshold = config["rebound_filter"]["threshold"] 
        distance = config["rebound_filter"]["distance"]
    except Exception as e:
        print(f"Using default config: {e}")
        offset, threshold, distance = 0.0, 0.1, 10
    
    # Process z-positions for rebound detection
    z_positions = positions[:, 2].copy()
    z_positions -= offset  # Apply ground offset
    z_positions *= -1      # Invert for peak finding (bounces = peaks)
    
    # Find peaks (rebounds/bounces)
    discontinuity_indices, _ = find_peaks(
        z_positions,
        height=threshold,
        distance=distance
    )
    
    # Truncate at first rebound if found
    if len(discontinuity_indices) > 0:
        first_rebound_idx = discontinuity_indices[0]
        return positions[:first_rebound_idx + 1]
    else:
        return positions


def first_hits_on_table_with_rebound_filter(file_path, shot_ids, 
                                           table_height=0.76, 
                                           max_hit_height=0.80, 
                                           apply_rebound_filter=True):
    """
    Enhanced version: finds ONLY first table hits, removes rebounded trajectories.
    """
    table_length, table_width = 2.74, 1.525
    hit_x, hit_y, hit_z, hit_ids = [], [], [], []
    launch_params = []
    
    rebound_config = {
        "rebound_filter": {
            "offset": 0.0,      # Ground level
            "threshold": 0.1,   # 10cm threshold for rebound detection
            "distance": 10      # Minimum 10 points between bounces
        }
    }

    with h5py.File(file_path, "r", libver="earliest", swmr=True) as f:
        traj_group = f["originals"]
        
        print(f"Processing {len(shot_ids)} trajectories with rebound filter...")
        filtered_count = 0
        
        for sid in shot_ids:
            key = f"{sid:05d}"
            if key not in traj_group:
                continue

            positions = traj_group[key]["positions"][:]
            if positions.shape[0] < 2:
                continue
                
            # APPLY REBOUND FILTER - This removes second bounces!
            if apply_rebound_filter:
                original_length = len(positions)
                positions = filter_trajectory_before_rebound(positions, rebound_config)
                if len(positions) < original_length:
                    filtered_count += 1
            
            # Store launch parameters
            try:
                if "launch_param" in traj_group[key]:
                    params = traj_group[key]["launch_param"][:]
                    launch_params.append(params)
                else:
                    launch_params.append(None)
            except:
                launch_params.append(None)

            # Find first table hit in filtered trajectory
            for i in range(1, len(positions)):
                z_curr = positions[i, 2]
                x_curr = positions[i, 0]
                y_curr = positions[i, 1]
                
                # Check if point is on table surface
                if (table_height <= z_curr <= max_hit_height and 
                    0 <= x_curr <= table_length and
                    -table_width/2 <= y_curr <= table_width/2):
                    
                    hit_x.append(x_curr)
                    hit_y.append(y_curr)
                    hit_z.append(z_curr)
                    hit_ids.append(key)
                    break

        print(f"Rebound filter applied to {filtered_count} trajectories")
        print(f"Found {len(hit_x)} first table hits (no rebounds)")

    if len(hit_x) == 0:
        print("No first hits found on the table.")
        return None

    return {
        'hit_ids': hit_ids,
        'hit_x': np.array(hit_x),
        'hit_y': np.array(hit_y), 
        'hit_z': np.array(hit_z),
        'launch_params': launch_params[:len(hit_x)],
        'filtered_trajectories': filtered_count
    }


def comprehensive_aimy_analysis(hits_data):
    """
    Complete analysis with parameter extraction and coordinate conversion
    """
    if hits_data is None:
        return
        
    hit_x = hits_data['hit_x']
    hit_y = hits_data['hit_y']
    hit_z = hits_data['hit_z']
    launch_params = hits_data['launch_params']
    
    print("=== COMPREHENSIVE AIMY DATASET ANALYSIS ===")
    print(f"📊 Total successful first hits: {len(hit_x)}")
    print(f"🔄 Trajectories filtered (rebounds removed): {hits_data['filtered_trajectories']}")
    print()
    
    # Extract launch parameters if available
    phi_vals, theta_vals, rpm_vals = [], [], []
    if launch_params and launch_params[0] is not None:
        for params in launch_params:
            if params is not None and len(params) >= 5:
                phi_vals.append(params[0])    # AIMY phi
                theta_vals.append(params[1])  # AIMY theta
                rpm_vals.append(params[2:5])  # RPMs
        
        phi_vals = np.array(phi_vals)
        theta_vals = np.array(theta_vals)
        
        print("🎯 LAUNCH PARAMETER ANALYSIS:")
        print(f"  • Phi range: {phi_vals.min():.3f} to {phi_vals.max():.3f} rad")
        print(f"  • Theta range: {theta_vals.min():.3f} to {theta_vals.max():.3f} rad")
        print(f"  • Phi coverage: {(phi_vals.max() - phi_vals.min()) / 0.8 * 100:.1f}% of AIMY range [0, 0.8]")
        print()
        
        # Convert AIMY phi to your coordinate system
        your_phi_vals = phi_vals - 0.4  # Apply conversion formula
        print("🔄 COORDINATE SYSTEM CONVERSION:")
        print(f"  • AIMY phi [{phi_vals.min():.3f}, {phi_vals.max():.3f}]")
        print(f"  • YOUR phi [{your_phi_vals.min():.3f}, {your_phi_vals.max():.3f}]")
        print(f"  • Conversion: your_phi = aimy_phi - 0.4")
        print()
    
    # Table coverage analysis
    table_length, table_width = 2.74, 1.525
    x_coverage = (hit_x.max() - hit_x.min()) / table_length * 100
    y_coverage = (hit_y.max() - hit_y.min()) / table_width * 100
    
    print("🏓 TABLE COVERAGE ANALYSIS:")
    print(f"  • X-coverage: {x_coverage:.1f}% of table length")
    print(f"  • Y-coverage: {y_coverage:.1f}% of table width")
    print(f"  • X-range: {hit_x.min():.3f} to {hit_x.max():.3f} m")
    print(f"  • Y-range: {hit_y.min():.3f} to {hit_y.max():.3f} m")
    print(f"  • Hit height range: {hit_z.min():.3f} to {hit_z.max():.3f} m")
    print()
    
    # Landing position statistics
    center_hits = np.sum(np.abs(hit_y) < 0.2)  # Within 20cm of center
    left_hits = np.sum(hit_y > 0.2)   # Left side
    right_hits = np.sum(hit_y < -0.2)  # Right side
    
    print("📍 LANDING POSITION DISTRIBUTION:")
    print(f"  • Center hits (|y| < 0.2m): {center_hits} ({center_hits/len(hit_y)*100:.1f}%)")
    print(f"  • Left side hits (y > 0.2m): {left_hits} ({left_hits/len(hit_y)*100:.1f}%)")
    print(f"  • Right side hits (y < -0.2m): {right_hits} ({right_hits/len(hit_y)*100:.1f}%)")
    print()


# MAIN EXECUTION
if __name__ == "__main__":
    file_path = r"sweep_dataset\MN5008_grid_data_all.hdf5"  # Update path
    shot_ids = list(range(942))  # Analyze first 942 shots
    
    hits = first_hits_on_table_with_rebound_filter(
        file_path, 
        shot_ids=shot_ids, 
        apply_rebound_filter=True
    )
    
    if hits:
        analysis = comprehensive_aimy_analysis(hits)
        print("✅ Analysis complete!")
    else:
        print("❌ No data found for analysis")
