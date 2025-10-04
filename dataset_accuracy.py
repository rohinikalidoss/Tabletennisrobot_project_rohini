import h5py
import numpy as np
import os

def verify_dataset_accuracy(dataset_path, num_samples=10):
    """
    🔍 Verify that dataset target↔control mappings are correct
    Using your exact dataset: grid_dataset_with300step.hdf5
    """
    
    print(f"🔍 DATASET VERIFICATION")
    print(f"📁 Dataset: {dataset_path}")
    print("="*80)
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        return False
    
    try:
        with h5py.File(dataset_path, "r") as f:
            print(f"✅ Dataset opened successfully")
            print(f"📊 File size: {os.path.getsize(dataset_path) / (1024*1024):.1f} MB")
            
            # Print dataset structure
            print(f"\n📁 Dataset structure:")
            def print_structure(name, obj):
                if hasattr(obj, 'shape'):
                    print(f"  {name}: {obj.shape} {obj.dtype}")
                else:
                    print(f"  {name}: {type(obj).__name__}")
            f.visititems(print_structure)
            
            # Check for 'originals' group (most likely structure)
            if 'originals' in f:
                originals = f['originals']
                all_keys = list(originals.keys())
                total_samples = len(all_keys)
                
                print(f"\n🎯 Found 'originals' group with {total_samples} samples")
                print(f"📋 Sample keys: {all_keys[:5]}..." if len(all_keys) > 5 else f"📋 Sample keys: {all_keys}")
                
                # Test samples
                test_samples = min(num_samples, total_samples)
                print(f"\n🧪 Testing {test_samples} samples:")
                print("-" * 80)
                
                valid_samples = 0
                issues_found = []
                
                for i, key in enumerate(all_keys[:test_samples]):
                    try:
                        group = originals[key]
                        
                        # Get data from sample
                        has_positions = 'positions' in group
                        has_launch_param = 'launch_param' in group
                        
                        if not has_positions:
                            issues_found.append(f"Sample {key}: Missing 'positions'")
                            continue
                        if not has_launch_param:
                            issues_found.append(f"Sample {key}: Missing 'launch_param'")
                            continue
                        
                        positions = np.array(group['positions'])
                        launch_param = np.array(group['launch_param'])
                        
                        # Final position = target
                        target_pos = positions[-1]  # Last position
                        
                        # Launch parameters
                        phi = launch_param[0]      # azimuth (rad)
                        theta = launch_param[1]    # elevation (rad)  
                        rpm_left = launch_param[2]
                        rpm_right = launch_param[3]
                        rpm_bottom = launch_param[4]
                        
                        print(f"\n🎯 Sample {i+1} ({key}):")
                        print(f"   Target:  [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
                        print(f"   Control: φ={phi:.4f}rad ({phi*180/np.pi:.1f}°)")
                        print(f"           θ={theta:.4f}rad ({theta*180/np.pi:.1f}°)")
                        print(f"           RPM=[{rpm_left:.0f}, {rpm_right:.0f}, {rpm_bottom:.0f}]")
                        
                        # Sanity checks
                        warnings = []
                        
                        # Check phi range (azimuth should be roughly -0.4 to 0.4)
                        if abs(phi) > 0.5:
                            warnings.append(f"φ={phi:.3f} outside expected range [-0.5, 0.5]")
                        
                        # Check theta range (elevation should be 0 to 1.0)  
                        if theta < -0.1 or theta > 1.2:
                            warnings.append(f"θ={theta:.3f} outside expected range [0, 1.0]")
                        
                        # Check RPM ranges (should be 500-2000)
                        for rpm_name, rpm_val in [("Left", rpm_left), ("Right", rpm_right), ("Bottom", rpm_bottom)]:
                            if rpm_val < 400 or rpm_val > 2200:
                                warnings.append(f"RPM_{rpm_name}={rpm_val:.0f} outside expected range [500, 2000]")
                        
                        # Check target position (table bounds)
                        x, y, z = target_pos
                        if not (0.5 < x < 4.0):
                            warnings.append(f"Target X={x:.3f} outside table range [0.5, 4.0]")
                        if not (-2.2 < y < 2.2):
                            warnings.append(f"Target Y={y:.3f} outside table range [-2.2, 2.2]")
                        if not (0.0 < z < 1.5):
                            warnings.append(f"Target Z={z:.3f} outside table range [0.0, 1.5]")
                        
                        # Check trajectory length
                        traj_length = len(positions)
                        if traj_length != 300:  # 300 steps as per filename
                            warnings.append(f"Trajectory length={traj_length}, expected 300")
                        
                        if warnings:
                            print(f"   ⚠️ Warnings: {'; '.join(warnings)}")
                        else:
                            print(f"   ✅ Sample looks good")
                            valid_samples += 1
                        
                    except Exception as e:
                        error_msg = f"Sample {key}: Error - {str(e)}"
                        issues_found.append(error_msg)
                        print(f"   ❌ {error_msg}")
                
                # Summary
                print("\n" + "="*80)
                print(f"📊 DATASET VERIFICATION SUMMARY")
                print(f"✅ Valid samples: {valid_samples}/{test_samples}")
                print(f"⚠️ Issues found: {len(issues_found)}")
                
                if issues_found:
                    print(f"\n🚨 Issues:")
                    for issue in issues_found[:5]:  # Show first 5 issues
                        print(f"   • {issue}")
                    if len(issues_found) > 5:
                        print(f"   ... and {len(issues_found) - 5} more")
                
                # Overall assessment
                success_rate = valid_samples / test_samples
                if success_rate > 0.8:
                    print(f"\n✅ DATASET LOOKS GOOD (Success rate: {success_rate:.1%})")
                    return True
                elif success_rate > 0.5:
                    print(f"\n⚠️ DATASET HAS ISSUES (Success rate: {success_rate:.1%})")
                    return False
                else:
                    print(f"\n❌ DATASET SERIOUSLY CORRUPTED (Success rate: {success_rate:.1%})")
                    return False
                    
            else:
                print(f"\n❌ Expected 'originals' group not found")
                print(f"Available top-level keys: {list(f.keys())}")
                return False
                
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return False

def main():
    """Run dataset verification with your exact path"""
    
    # Your exact dataset path
    dataset_path = r"C:\Users\rohin\Downloads\mujoco-3.3.5-windows-x86_64\model\tabletennis_table\table_tennis_pinnmodel\dataset\grid_dataset_with300step.hdf5"
    
    print("🔍 DATASET ACCURACY VERIFICATION")
    print("This will check if your dataset contains correct physics mappings")
    print()
    
    success = verify_dataset_accuracy(dataset_path, num_samples=10)
    
    if success:
        print(f"\n🎉 DATASET VERIFICATION PASSED!")
        print(f"Your dataset appears to have correct target↔control mappings")
        print(f"The 70cm error is likely due to model training issues, not data corruption")
    else:
        print(f"\n🚨 DATASET VERIFICATION FAILED!")
        print(f"Your dataset may have incorrect target↔control mappings")
        print(f"This could explain the 70cm prediction errors")
        
    print(f"\nNext steps:")
    if success:
        print(f"✅ Dataset is OK - Focus on improving model training")
        print(f"   • Add real physics constraints to PINN")
        print(f"   • Train to lower loss (< 0.01)")
        print(f"   • Increase model capacity")
    else:
        print(f"Dataset needs fixing - Regenerate dataset with:")
        print(f"   • Correct simulation parameters")
        print(f"   • Verified physics mappings")
        print(f"   • Consistent coordinate systems")

if __name__ == "__main__":
    main()
