# AIMY_TARGET_SHOOTING_WITH_SIMULATION_RANGES.py
# 🎯 YOUR EXACT WORKFLOW: AIMY NN → Theta Offset → MuJoCo → Ranges

import sys
import os
import h5py
import numpy as np
import json
import logging
import pathlib
from contextlib import redirect_stdout
from io import StringIO
from datetime import datetime
import pandas as pd

# Add AIMY target shooting to path (adjust as needed)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import AIMY classes
from aimy_target_shooting.configuration import get_config_path
from aimy_target_shooting.target_shooting_nn import TargetShootingNN

# Import your MuJoCo simulation
from launch_fixed_magnusforce import TrajectoryMuJoCoIntegration

class AimyWithSimulationAccuracyTester:
    def __init__(self, model_path, scaling_path):
        print("🎯 AIMY TARGET SHOOTING NN + SIMULATION + RANGES")
        print("=" * 60)
        
        # Load AIMY TargetShootingNN 
        self.nn = TargetShootingNN({}, verbose=False)
        self.nn.load_model(model_path)
        self.nn.load_scaling(scaling_path)
        print("✅ AIMY TargetShootingNN loaded!")
        
        # Load MuJoCo simulation
        try:
            with redirect_stdout(StringIO()):
                self.physics_sim = TrajectoryMuJoCoIntegration('radial_device.xml')
            print("✅ MuJoCo simulation loaded!")
            self.physics_available = True
        except Exception as e:
            print(f"❌ MuJoCo failed: {e}")
            self.physics_available = False

    def your_exact_workflow_with_ranges(self, target_position):
        """🎯 YOUR EXACT WORKFLOW: Target → AIMY NN → Offset → Simulation → Ranges"""
        
        target_x, target_y, target_z = target_position
        
        # STEP 1: GET PREDICTIONS FROM AIMY NN (YOUR EXACT CODE)
        control_parameters = self.nn.compute_control_parameters(target_position)
        
        phi_predicted = round(float(control_parameters[0]), 2)
        theta_predicted = round(float(control_parameters[1]), 2)
        rpm_left = round(float(control_parameters[2]), 2)
        rpm_right = round(float(control_parameters[3]), 2)
        rpm_bottom = round(float(control_parameters[4]), 2)
        
        print(f"🎯 Target: [{target_x}, {target_y}, {target_z}]")
        print(f"🤖 AIMY Predicted: Phi={phi_predicted}, Theta={theta_predicted}")
        print(f"   RPMs: [{rpm_left}, {rpm_right}, {rpm_bottom}]")
        
        # STEP 2: APPLY THETA OFFSET (2 DECIMAL PLACES)
        theta_offset = 0.15  # Your standard offset
        theta_final = round(theta_predicted - theta_offset, 2)
        print(f"⚙️  Theta Offset: {theta_predicted} - {theta_offset} = {theta_final}")
        
        # STEP 3: RUN MUJOCO SIMULATION
        if self.physics_available:
            try:
                with redirect_stdout(StringIO()):
                    # Set parameters in simulation (using your offset theta!)
                    self.physics_sim.phi = phi_predicted
                    self.physics_sim.theta = theta_final  # WITH OFFSET!
                    self.physics_sim.rpm_spin = [rpm_left, rpm_right, rpm_bottom]
                    
                    # Calculate initial conditions
                    v_linear, w_vector = self.physics_sim.calculate_initial_velocity_and_spin()
                    initial_pos = np.array([-0.5, 0.0, 1.25], dtype=float)
                    
                    # Launch and record
                    self.physics_sim.first_table_hit = None
                    self.physics_sim.launch_ball_in_mujoco(initial_pos, v_linear, spin=w_vector)
                    positions, times, velocities = self.physics_sim.record_full_trajectory(max_time=3.0)
                
                print(f"🎮 Simulation: phi={phi_predicted}, theta={theta_final}, RPMs=[{rpm_left}, {rpm_right}, {rpm_bottom}]")
                
                # STEP 4: GET TABLE HIT AND CALCULATE DEVIATION
                if hasattr(self.physics_sim, 'first_table_hit') and self.physics_sim.first_table_hit:
                    _, hit_pos, _, _ = self.physics_sim.first_table_hit
                    actual_x = round(hit_pos[0], 3)
                    actual_y = round(hit_pos[1], 3)
                    actual_z = round(hit_pos[2], 3)
                    
                    print(f"🏓 Hit Position: [{actual_x}, {actual_y}, {actual_z}]")
                    
                    # Calculate deviation from target
                    deviation_m = np.sqrt((actual_x - target_x)**2 + (actual_y - target_y)**2)
                    deviation_cm = deviation_m * 100
                    
                    print(f"📏 Deviation: {deviation_cm:.1f}cm from target")
                    
                    # STEP 5: ASSIGN TO ACCURACY RANGES
                    if deviation_cm <= 5:
                        accuracy_range = "≤5cm EXCELLENT"
                        status = "🎯"
                    elif deviation_cm <= 10:
                        accuracy_range = "5-10cm GOOD"
                        status = "✅"
                    elif deviation_cm <= 15:
                        accuracy_range = "10-15cm MODERATE"
                        status = "⚠️"
                    else:
                        accuracy_range = ">15cm POOR"
                        status = "❌"
                    
                    print(f"📊 Range: {accuracy_range} {status}")
                    
                    return {
                        'target': target_position,
                        'aimy_prediction': [phi_predicted, theta_predicted, rpm_left, rpm_right, rpm_bottom],
                        'final_params': [phi_predicted, theta_final, rpm_left, rpm_right, rpm_bottom],
                        'hit_position': [actual_x, actual_y, actual_z],
                        'deviation_cm': round(deviation_cm, 1),
                        'accuracy_range': accuracy_range,
                        'status': status,
                        'success': True
                    }
                else:
                    print("❌ Ball missed the table!")
                    return {
                        'target': target_position,
                        'aimy_prediction': [phi_predicted, theta_predicted, rpm_left, rpm_right, rpm_bottom],
                        'final_params': [phi_predicted, theta_final, rpm_left, rpm_right, rpm_bottom],
                        'hit_position': None,
                        'deviation_cm': 999.9,
                        'accuracy_range': "MISS",
                        'status': "❌",
                        'success': False
                    }
                    
            except Exception as e:
                print(f"❌ Simulation error: {e}")
                return {
                    'target': target_position,
                    'error': str(e),
                    'success': False
                }
        else:
            print("❌ Physics simulation not available")
            return {'error': 'No physics simulation', 'success': False}

    def test_multiple_positions_with_ranges(self, test_positions):
        """🎯 Test multiple positions and generate range statistics"""
        
        print(f"\n🎯 TESTING {len(test_positions)} POSITIONS WITH AIMY NN + SIMULATION:")
        print("=" * 80)
        
        all_results = []
        successful_results = []
        
        for i, target in enumerate(test_positions, 1):
            print(f"\n--- Position {i:2d}/{len(test_positions)} ---")
            
            result = self.your_exact_workflow_with_ranges(target)
            all_results.append(result)
            
            if result.get('success', False):
                successful_results.append(result)
                print(f"✅ Success: {result['deviation_cm']:.1f}cm {result['status']}")
            else:
                print(f"❌ Failed")
        
        # Generate comprehensive statistics
        self.generate_range_statistics(all_results, successful_results)
        
        return all_results, successful_results

    def generate_range_statistics(self, all_results, successful_results):
        """📊 Generate detailed accuracy statistics by ranges"""
        
        print(f"\n📊 AIMY NN + SIMULATION ACCURACY STATISTICS:")
        print("=" * 60)
        
        total_tests = len(all_results)
        successful_hits = len(successful_results)
        
        print(f"Total tests: {total_tests}")
        print(f"Successful hits: {successful_hits} ({successful_hits/total_tests*100:.1f}%)")
        print(f"Misses: {total_tests - successful_hits} ({(total_tests - successful_hits)/total_tests*100:.1f}%)")
        
        if successful_results:
            deviations = [r['deviation_cm'] for r in successful_results]
            
            print(f"\n🎯 DEVIATION STATISTICS:")
            print("-" * 30)
            print(f"  Mean error: {np.mean(deviations):.1f}cm")
            print(f"  Median error: {np.median(deviations):.1f}cm")
            print(f"  Best shot: {np.min(deviations):.1f}cm")
            print(f"  Worst shot: {np.max(deviations):.1f}cm")
            print(f"  Std deviation: {np.std(deviations):.1f}cm")
            
            # Count by accuracy ranges
            excellent = len([d for d in deviations if d <= 5])
            good = len([d for d in deviations if 5 < d <= 10])
            moderate = len([d for d in deviations if 10 < d <= 15])
            poor = len([d for d in deviations if d > 15])
            
            print(f"\n🏆 ACCURACY RANGES (out of {successful_hits} hits):")
            print("-" * 50)
            print(f" 🎯 ≤5cm (EXCELLENT):     {excellent:2d} ({excellent/successful_hits*100:5.1f}%)")
            print(f" ✅ 5-10cm (GOOD):        {good:2d} ({good/successful_hits*100:5.1f}%)")
            print(f" ⚠️  10-15cm (MODERATE):   {moderate:2d} ({moderate/successful_hits*100:5.1f}%)")
            print(f" ❌ >15cm (POOR):         {poor:2d} ({poor/successful_hits*100:5.1f}%)")
            
            # Overall accuracy assessment
            excellent_rate = excellent / total_tests * 100
            good_plus_rate = (excellent + good) / total_tests * 100
            
            print(f"\n🎖️  OVERALL ASSESSMENT:")
            print("-" * 25)
            if excellent_rate >= 50:
                print(f" 🏆 OUTSTANDING: {excellent_rate:.1f}% excellent shots!")
            elif good_plus_rate >= 60:
                print(f" 🥇 VERY GOOD: {good_plus_rate:.1f}% good+ shots!")
            elif good_plus_rate >= 40:
                print(f" 🥉 ACCEPTABLE: {good_plus_rate:.1f}% good+ shots")
            else:
                print(f" 📈 NEEDS IMPROVEMENT: Only {good_plus_rate:.1f}% good+ shots")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            df = pd.DataFrame(all_results)
            filename = f"AIMY_simulation_results_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"\n💾 Results saved: {filename}")


def main():
    """🎯 Main function to test your AIMY NN with simulation"""
    
    # Initialize with YOUR AIMY model paths
    tester = AimyWithSimulationAccuracyTester(
        model_path=r"C:\tmp\nn_model\model12\model12.hdf5",
        scaling_path=r"C:\tmp\nn_model\model12"
    )
    
    if tester.physics_available:
        # Test positions (adjust as needed)
        test_positions = [
            np.array([2.3, 0.6, 0.784]),  # Your example
            np.array([2.2, 0.7, 0.784]),
            np.array([2.4, 0.5, 0.784]),
            np.array([2.1, 0.8, 0.784]),
            np.array([2.5, 0.6, 0.784]),
            np.array([2.0, 0.7, 0.784]),
            np.array([2.3, 0.4, 0.784]),
            np.array([2.6, 0.7, 0.784]),
            np.array([2.2, 0.9, 0.784]),
            np.array([2.4, 0.8, 0.784]),
        ]
        
        # Run the complete workflow
        all_results, successful_results = tester.test_multiple_positions_with_ranges(test_positions)
        
        print(f"\n✅ COMPLETE! Your AIMY NN workflow with simulation ranges is done!")
        print(f"🎯 Check how many shots fell into each accuracy range!")
        
    else:
        print("❌ Cannot test - physics simulation not available")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,  # Suppress most logs
        format="%(asctime)s %(levelname)-6s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    np.set_printoptions(suppress=True)
    main()
