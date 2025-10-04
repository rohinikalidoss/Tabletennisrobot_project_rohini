# launch.py
#fixed an magunus force for y 

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time
import sys


class TrajectoryMuJoCoIntegration:
    def __init__(self, xml_file: str):
        # Load model & data
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        # IDs
        try:
            self.ball_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        except Exception:
            self.ball_joint_id = None
        self.ball_geom_name = "ping_pong_ball"
        self.table_geom_name = "table_surface"
        self.floor_geom_name = "court_floor"
        self.marker_geom_name = "marker"
        def safe_geom_id(name):
            try:
                return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            except Exception:
                return -1
        self.ball_geom_id = safe_geom_id(self.ball_geom_name)
        self.table_geom_id = safe_geom_id(self.table_geom_name)
        self.floor_geom_id = safe_geom_id(self.floor_geom_name)
        self.marker_id = safe_geom_id(self.marker_geom_name)
        if self.marker_id == -1:
            self.marker_id = None
        # Table constants
        self.table_height = 0.76
        self.table_half_x = 1.37
        self.table_half_y = 0.76
        self.g = 9.81
        
        self.phi = 0
        self.theta = 0.9
        self.rpm_spin = [ 1500,900,600]

        self.wheel_radius = 0.05
        self.first_table_hit = None   
        self.first_ground_hit = None  
        self.bounce_points = []       
        self.last_launch = None      

    def rpm_to_radsec(self, rpm):
        return rpm * 2.0 * np.pi / 60
    
    def rpm_triplet_to_speed(self, rpm_triplet):   
        R = self.wheel_radius
        v_left = self.rpm_to_radsec(rpm_triplet[0]) * R
        v_right = self.rpm_to_radsec(rpm_triplet[1]) * R
        v_bottom = self.rpm_to_radsec(rpm_triplet[2]) * R
        speed = max(0.0, (abs(v_left) + abs(v_right) + abs(v_bottom)) / 3.0)
        return float(speed), (v_left, v_right, v_bottom)
    
    def spherical_to_velocity(self, phi, theta, speed):
        vx = speed * np.cos(theta) * np.cos(phi)
        vy = speed * np.cos(theta) * np.sin(phi)
        vz = speed * np.sin(theta)
        return np.array([vx, vy, vz], dtype=float)
    # -------------------------
    # Launch & spin application
    # -------------------------
    def calculate_initial_velocity_and_spin(self):
     speed, tangentials = self.rpm_triplet_to_speed(self.rpm_spin)
     v_linear = self.spherical_to_velocity(self.phi, self.theta, speed)

    # Version 2: Physics-based spin calculation
     rpm_tl = self.rpm_spin[0]  # Left wheel
     rpm_tr = self.rpm_spin[1]  # Right wheel
     rpm_bc = self.rpm_spin[2]  # Bottom wheel
    

     rpm_to_rads = 2 * np.pi / 60
     front_avg = (rpm_tl + rpm_tr) / 2
     spin_pitch = (rpm_tr - rpm_tl) * rpm_to_rads * 0.6  # Topspin/backspin

    # FIXED SIGN
     spin_yaw = (rpm_bc - front_avg) * rpm_to_rads * 0.7  # Yaw spin (side curve)

     spin_roll = 0.0
     wx = spin_roll
     wy = spin_yaw
     wz = -spin_pitch
     print(f"Yaw spin (wy): {wy:.3f} | Expected: +wy = LEFT, -wy = RIGHT")


     """#  DEBUG PRINT HERE
     print(f"\n=== SPIN CALCULATION DEBUG ===")
     print(f"RPM Values: Left={rpm_tl}, Right={rpm_tr}, Bottom={rpm_bc}")
     print(f"Yaw spin (wy): {wy:.3f}")
     print("Expected: +wy = ball curves LEFT, -wy = ball curves RIGHT")
     print(f"Spin vector: [{wx:.3f}, {wy:.3f}, {wz:.3f}]")
     print("=" * 40)"""

     w_vector = np.array([wx, wy, wz], dtype=float)
     return v_linear, w_vector

    def launch_ball_in_mujoco(self, initial_pos, initial_vel, spin=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.time = 0.0
        self.first_table_hit = None
        self.first_ground_hit = None
        self.bounce_points = []
        self.last_launch = None
        if self.ball_joint_id is not None:
            qposadr = self.model.jnt_qposadr[self.ball_joint_id]
            qveladr = self.model.jnt_dofadr[self.ball_joint_id]
            self.data.qpos[qposadr:qposadr+3] = np.array(initial_pos, dtype=float)
            self.data.qpos[qposadr+3:qposadr+7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            self.data.qvel[qveladr:qveladr+3] = np.array(initial_vel, dtype=float)
            if spin is not None:
                self.data.qvel[qveladr+3:qveladr+6] = np.array(spin, dtype=float)

                lf, rf, bf = self.rpm_to_launcher_forces()
                self.apply_launcher_forces(left_force=lf, right_force=rf, bottom_force=bf)
            else:
                self.data.qvel[qveladr+3:qveladr+6] = np.zeros(3, dtype=float)
        else:
            self.data.qpos[0:3] = np.array(initial_pos, dtype=float)
            self.data.qvel[0:3] = np.array(initial_vel, dtype=float)
            if spin is not None:
                self.data.qvel[3:6] = np.array(spin, dtype=float)

        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step1(self.model, self.data)
        self.last_launch = (np.array(initial_pos, dtype=float),
                            np.array(initial_vel, dtype=float),
                            np.array(spin, dtype=float) if spin is not None else None)
        print("\nBall launched:")
        print(f"  pos={initial_pos}, vel={initial_vel}")
        print(f"  spin={spin}")

    # -------------------------
    # Hitpoint marking
    # -------------------------
    def mark_hitpoint_in_viewer(self, pos):
        """Move the 'marker' geom to the hit position if marker exists."""
        if self.marker_id is not None:
            try:
                self.data.geom_xpos[self.marker_id] = np.array(pos, dtype=float)
                mujoco.mj_forward(self.model, self.data)
            except Exception:
                pass

    def calculate_magnus_force(self, velocity, angular_velocity):
     velocity = np.array(velocity, dtype=float)
     angular_velocity = np.array(angular_velocity, dtype=float)
     speed = np.linalg.norm(velocity)
    
     if speed < 0.1:
        return np.zeros(3)
    
    # CORRECT: Use proper cross product order
     cross_product = np.cross(angular_velocity, velocity)  # ω × v
     cross_magnitude = np.linalg.norm(cross_product)
    
     if cross_magnitude > 0:
        air_density = 1.225
        ball_radius = 0.02
        magnus_coefficient = 0.05
        ball_area = np.pi * ball_radius**2
        force_magnitude = magnus_coefficient * air_density * ball_area * cross_magnitude
        force_direction = cross_product / cross_magnitude
        magnus_force = force_magnitude * force_direction
       
     else:
        magnus_force = np.zeros(3)
    
    # NO MANUAL Y-FLIP NEEDED with correct physics
     return magnus_force


 
    
    def get_ball_body_id(self):
     try:
        ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ping_pong_ball")
        return ball_body_id
     except:
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        except:
            try:
                return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Ball")
            except:
                print("Error: Could not find ball body in model")
                return None
            
    def apply_magnus_force(self):
     ball_body_id = self.get_ball_body_id()
     if ball_body_id is None:
        return
    
     qposadr = self.model.jnt_qposadr[self.ball_joint_id] if self.ball_joint_id is not None else 0
     qveladr = self.model.jnt_dofadr[self.ball_joint_id] if self.ball_joint_id is not None else 0
    
     velocity = self.data.qvel[qveladr:qveladr+3]
     angular_velocity = self.data.qvel[qveladr+3:qveladr+6]
    
     magnus_force = self.calculate_magnus_force(velocity, angular_velocity)

     magnus_force *= 1
     magnus_force[1]=-magnus_force[1]
     self.data.qfrc_applied[6*ball_body_id:6*ball_body_id+3] = magnus_force


    # DEBUG: Print every 100 steps to avoid spam
     if hasattr(self, 'debug_counter'):
        self.debug_counter += 1
     else:
        self.debug_counter = 1
        
     if self.debug_counter % 100 == 0 and np.linalg.norm(magnus_force) > 0.001:
        self.debug_magnus_force(velocity, angular_velocity, magnus_force)
     self.data.qfrc_applied[6*ball_body_id:6*ball_body_id+3] = magnus_force
     """
     if np.linalg.norm(magnus_force) > 0.01:
        print(f"Magnus force applied: {magnus_force}")
     print(f"Velocity: {velocity}")
     print(f"Angular velocity: {angular_velocity}")  
     print(f"Magnus force: {magnus_force}")
     print(f"Magnus Y-component: {magnus_force[1]} (- = right, + = left)")"""

    




    # -------------------------
    # Trajectory recording
    # -------------------------
    def record_full_trajectory(self, max_time=4.0):
    
     positions = []
     velocities = []
     times = []

     qposadr = self.model.jnt_qposadr[self.ball_joint_id] if self.ball_joint_id is not None else 0
     qveladr = self.model.jnt_dofadr[self.ball_joint_id] if self.ball_joint_id is not None else 0
     
     launch_duration=0.01
     while self.data.time < max_time:
        self.apply_magnus_force()

       

        # Step simulation
        mujoco.mj_step(self.model, self.data)
        pos = self.data.qpos[qposadr:qposadr+3].copy()
        vel = self.data.qvel[qveladr:qveladr+3].copy()
        positions.append(pos)
        velocities.append(vel)
        times.append(self.data.time)
        ncon = int(getattr(self.data, "ncon", 0))
        for i in range(ncon):
            try:
                c = self.data.contact[i]
            except IndexError:
                continue
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            ball_involved = (g1 == self.ball_geom_id) or (g2 == self.ball_geom_id)
            if not ball_involved:
                continue
            other = g2 if g1 == self.ball_geom_id else g1
            if self.table_geom_id != -1 and other == self.table_geom_id and self.first_table_hit is None:
                pos_hit = c.pos.copy()
                vel_hit = vel.copy()
                spin_hit = self.data.qvel[qveladr+3:qveladr+6].copy()
                self.first_table_hit = (self.data.time, pos_hit, vel_hit, spin_hit)
                self.bounce_points.append(pos_hit)
                print(f"Table Hit: {pos_hit}, Velocity: {vel_hit}, Spin: {spin_hit}")
                self.mark_hitpoint_in_viewer(pos_hit)
                positions.append(pos_hit)
                print(f"afetr :{pos_hit}")

            if self.floor_geom_id != -1 and other == self.floor_geom_id and self.first_ground_hit is None:
                pos_hit = c.pos.copy()
                vel_hit = vel.copy()
                spin_hit = self.data.qvel[qveladr+3:qveladr+6].copy()
                self.first_ground_hit = (self.data.time, pos_hit, vel_hit, spin_hit)
                self.bounce_points.append(pos_hit)
                print(f"Ground Hit: {pos_hit}, Velocity: {vel_hit}, Spin: {spin_hit}")
                self.mark_hitpoint_in_viewer(pos_hit)
        if pos[2] < 0.01 or abs(pos[0]) > 8.0:
            break
     return np.array(positions), np.array(times), np.array(velocities)

    # -------------------------
    # Analysis + plotting
    # -------------------------
    def analyze_single_bounce_drill(self, positions):
        if positions.size == 0:
            print("No recorded positions.")
            return
        if self.last_launch:
            pos0, vel0, spin0 = self.last_launch
            print("\n" + "=" * 50)
            print(" LAUNCH INFO ")
            print("=" * 50)
            print(f"Launch Position: X={pos0[0]:.3f}, Y={pos0[1]:.3f}, Z={pos0[2]:.3f}")
            print(f"Launch Velocity: Vx={vel0[0]:.3f}, Vy={vel0[1]:.3f}, Vz={vel0[2]:.3f}")
            if spin0 is not None:
                print(f"Launch Spin (rad/s): {spin0}")
            print("=" * 50)
        if self.first_table_hit is not None:
            _, pos_t, vel_t, spin_t = self.first_table_hit
            print("\n" + "=" * 50)
            print(" TABLE HIT (first) ")
            print("=" * 50)
            print(f"Hit Position: X={pos_t[0]:.3f}, Y={pos_t[1]:.3f}, Z={pos_t[2]:.3f}")
            print(f"Velocity: Vx={vel_t[0]:.3f}, Vy={vel_t[1]:.3f}, Vz={vel_t[2]:.3f}")
            print(f"Spin (rad/s): Wx={spin_t[0]:.3f}, Wy={spin_t[1]:.3f}, Wz={spin_t[2]:.3f}")
            print("=" * 50)
        if self.first_ground_hit is not None:
            _, pos_g, vel_g, spin_g = self.first_ground_hit
            print("\n" + "=" * 50)
            print(" GROUND HIT (first) ")
            print("=" * 50)
            print(f"Hit Position: X={pos_g[0]:.3f}, Y={pos_g[1]:.3f}, Z={pos_g[2]:.3f}")
            print(f"Velocity: Vx={vel_g[0]:.3f}, Vy={vel_g[1]:.3f}, Vz={vel_g[2]:.3f}")
            print(f"Spin (rad/s): Wx={spin_g[0]:.3f}, Wy={spin_g[1]:.3f}, Wz={spin_g[2]:.3f}")
            print("=" * 50)

    def plot_trajectory(self, positions, times):
     if positions.size == 0:
        print("No data to plot.")
        return

     plt.figure(figsize=(15, 6))

    # Side view (X-Z) with trajectory line
     plt.subplot(1, 3, 1)
     plt.plot(positions[:, 0], positions[:, 2], '-b', linewidth=2, marker='o', markersize=3, label='Ball path')
    
    # Add direction arrows
     for i in range(0, len(positions)-1, len(positions)//5):
        dx = positions[i+1, 0] - positions[i, 0]
        dz = positions[i+1, 2] - positions[i, 2]
        plt.arrow(positions[i, 0], positions[i, 2], dx*0.3, dz*0.3,
                  head_width=0.05, head_length=0.05, fc='red', ec='red')
    
     plt.axhline(self.table_height, color='k', linewidth=3, label='table top')
     plt.axvline(0.0, color='gray', linestyle='--', label='net')
     plt.xlabel('X (m)')
     plt.ylabel('Z (m)')
     plt.title('Side view (X-Z) - Ball Trajectory')
     plt.grid(True, alpha=0.3)
     plt.legend()

    # Top view (X-Y) with trajectory line - KEEP ORIGINAL (NO INVERSION)
     plt.subplot(1, 3, 2)
     plt.plot(positions[:, 0], positions[:, 1], '-r', linewidth=2, marker='o', markersize=3, label='Ball path')
    
    # Add direction arrows - KEEP ORIGINAL
     for i in range(0, len(positions)-1, len(positions)//5):
        dx = positions[i+1, 0] - positions[i, 0]
        dy = positions[i+1, 1] - positions[i, 1]  # KEEP ORIGINAL
        plt.arrow(positions[i, 0], positions[i, 1], dx*0.3, dy*0.3,
                  head_width=0.05, head_length=0.05, fc='blue', ec='blue')

    # Mark bounce points - KEEP ORIGINAL
     if self.bounce_points:
        pts = np.array(self.bounce_points)
        plt.scatter(pts[:, 0], pts[:, 1], color='green', marker='x', s=150,
                    linewidth=3, label='Bounce points')

     plt.xlabel('X (m)')
     plt.ylabel('Y (m) [Negative=Right, Positive=Left]')
     plt.title('Top view (X-Y) - Ball Deviation [CORRECTED]')
     plt.grid(True, alpha=0.3)
     plt.axis('equal')
     plt.legend()

    # 3D trajectory plot - KEEP ORIGINAL
     ax = plt.subplot(1, 3, 3, projection='3d')
     ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-g', linewidth=2, marker='o', markersize=2)
     ax.set_xlabel('X (m)')
     ax.set_ylabel('Y (m)')
     ax.set_zlabel('Z (m)')
     ax.set_title('3D Trajectory')

     plt.tight_layout()
     plt.show()


    # -------------------------
    # Drill / Passive modes
    # -------------------------
    def setup_robot_drill(self):
        x0, y0, z0 = -0.5,0.0,1.25
        target_x = 0.35
        target_y = 0.0
        target_z = self.table_height
        t_flight = 0.75
        return x0, y0, z0, target_x, target_y, target_z, t_flight

    def calculate_trajectory(self, x0, y0, z0, target_x, target_y, target_z, t_flight):
        speed, _ = self.rpm_triplet_to_speed(self.rpm_spin)
        v0 = self.spherical_to_velocity(self.phi, self.theta, speed)
        return v0

    def run_robot_drill(self, num_balls=3, ball_interval=2.0):
        print("Starting drill...")
        x0, y0, z0, target_x, target_y, target_z, t_flight = self.setup_robot_drill()
        _, spin_vector = self.calculate_initial_velocity_and_spin()
        spin_rad = spin_vector 
        for i in range(num_balls):
            print(f"\n--- Ball {i+1}/{num_balls} ---")
            tx = target_x 
            ty = target_y 

            v0 = self.calculate_trajectory(x0, y0, z0, tx, ty, target_z, t_flight)
            _, spin_vector = self.calculate_initial_velocity_and_spin()

            self.launch_ball_in_mujoco(np.array([x0, y0, z0], dtype=float), v0, spin=spin_vector)
            positions, times, velocities = self.record_full_trajectory()
            self.analyze_single_bounce_drill(positions)
            self.plot_trajectory(positions, times)

            if i < num_balls - 1:
                time.sleep(ball_interval)

    def run_passive_simulation(self):
    
     reset_cooldown = 2.0
     last_reset_time = 0.0
     wait_steps = 200
     step_count = 0
     launched = False

     print("Launching passive viewer...")
     viewer = mujoco.viewer.launch_passive(self.model, self.data)
     print("Passive mode started. Close the viewer to exit.")
     while viewer.is_running():
        if launched:
            self.apply_magnus_force()
            if self.data.time < 0.01:  # wheels only act right after launch
              lf, rf, bf = self.rpm_to_launcher_forces()
              self.apply_launcher_forces(left_force=lf, right_force=rf, bottom_force=bf)
        mujoco.mj_step(self.model, self.data)
        time.sleep(self.model.opt.timestep * 4.0)
        if not launched and step_count >= wait_steps:
            x0, y0, z0, target_x, target_y, target_z, t_flight = self.setup_robot_drill()
            v0 = self.calculate_trajectory(x0, y0, z0, target_x, target_y, target_z, t_flight)
            _, spin_vector = self.calculate_initial_velocity_and_spin()
            self.launch_ball_in_mujoco(np.array([x0, y0, z0], dtype=float), v0, spin=spin_vector)
            launched = True
            last_reset_time = time.time()
            print("Ball launched!")
        ncon = int(getattr(self.data, "ncon", 0))
        for i in range(ncon):
            try:
                c = self.data.contact[i]
            except IndexError:
                continue 
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            ball_involved = (g1 == self.ball_geom_id) or (g2 == self.ball_geom_id)
            if not ball_involved:
                continue
            other = g2 if g1 == self.ball_geom_id else g1

            # Table hit
            if self.table_geom_id != -1 and other == self.table_geom_id and self.first_table_hit is None:
                pos_hit = self.data.qpos[self.model.jnt_qposadr[self.ball_joint_id]:self.model.jnt_qposadr[self.ball_joint_id]+3].copy()
                vel_hit = self.data.qvel[self.model.jnt_dofadr[self.ball_joint_id]:self.model.jnt_dofadr[self.ball_joint_id]+3].copy()
                spin_hit = self.data.qvel[self.model.jnt_dofadr[self.ball_joint_id]+3:self.model.jnt_dofadr[self.ball_joint_id]+6].copy()
                self.first_table_hit = (self.data.time, pos_hit, vel_hit, spin_hit)
                self.bounce_points.append(pos_hit)
                print(f"\nTable Hit: Position={pos_hit}, Velocity={vel_hit}, Spin={spin_hit}")
                self.mark_hitpoint_in_viewer(pos_hit)

            # Ground hit
            if self.floor_geom_id != -1 and other == self.floor_geom_id and self.first_ground_hit is None:
                pos_hit = self.data.qpos[self.model.jnt_qposadr[self.ball_joint_id]:self.model.jnt_qposadr[self.ball_joint_id]+3].copy()
                vel_hit = self.data.qvel[self.model.jnt_dofadr[self.ball_joint_id]:self.model.jnt_dofadr[self.ball_joint_id]+3].copy()
                spin_hit = self.data.qvel[self.model.jnt_dofadr[self.ball_joint_id]+3:self.model.jnt_dofadr[self.ball_joint_id]+6].copy()
                self.first_ground_hit = (self.data.time, pos_hit, vel_hit, spin_hit)
                self.bounce_points.append(pos_hit)
                print(f"\nGround Hit: Position={pos_hit}, Velocity={vel_hit}, Spin={spin_hit}")
                self.mark_hitpoint_in_viewer(pos_hit)

        # Auto-reset when ball is lost
        qposadr = self.model.jnt_qposadr[self.ball_joint_id] if self.ball_joint_id is not None else 0
        pos = self.data.qpos[qposadr:qposadr+3]
        if (time.time() - last_reset_time > reset_cooldown and (pos[2] < 0.1 or abs(pos[0]) > 8.0)):
            print("Auto-resetting ball...")
            if self.last_launch:
                init_pos, init_vel, spin = self.last_launch
                self.launch_ball_in_mujoco(init_pos, init_vel, spin)
                last_reset_time = time.time()

        step_count += 1
        viewer.sync()

    def debug_magnus_force(self, velocity, angular_velocity, magnus_force):
     print(f"\n=== MAGNUS FORCE DEBUG ===")
     print(f"Velocity: [{velocity[0]:.3f}, {velocity[1]:.3f}, {velocity[2]:.3f}]")
     print(f"Angular Velocity: [{angular_velocity[0]:.3f}, {angular_velocity[1]:.3f}, {angular_velocity[2]:.3f}]")
     print(f"Magnus Force: [{magnus_force[0]:.6f}, {magnus_force[1]:.6f}, {magnus_force[2]:.6f}]")
     print(f"Magnus Y-component: {magnus_force[1]:.6f}")
     print(f"Force Direction: {'RIGHT' if magnus_force[1] < 0 else 'LEFT'} (assuming MuJoCo Y-left convention)")
     print("=" * 40)

    def apply_launcher_forces(self, left_force=0.0, right_force=0.0, bottom_force=0.0):
      """
     Apply external forces that simulate the 3-wheel launcher.
     Left wheel: pushes ball in +Y (leftward)
     Right wheel: pushes ball in -Y (rightward)
     Bottom wheel: pushes ball upward (+Z)
     """
      ball_body_id = self.get_ball_body_id()
      if ball_body_id is None:
        return

      force = np.array([
        0.0,                            # X handled by initial velocity
        left_force - right_force,       # Net side force
        bottom_force                    # Lift
     ], dtype=float)

      self.data.qfrc_applied[6*ball_body_id:6*ball_body_id+3] += force

      if np.linalg.norm(force) > 1e-6:
        print(f"[Launcher Force] Applied: {force}")

    def rpm_to_launcher_forces(self):
         ball_body_id = self.get_ball_body_id()
         if ball_body_id is None:
            return 0.0, 0.0, 0.0

         mass = self.model.body_mass[ball_body_id]
         dt   = self.model.opt.timestep
         scaling=0.02

         def rpm_to_force(rpm):
            v = self.rpm_to_radsec(rpm) * self.wheel_radius   # tangential speed
            return mass * v / dt*scaling              # approximate impulse

         left_force   = rpm_to_force(self.rpm_spin[0])
         right_force  = rpm_to_force(self.rpm_spin[1])
         bottom_force = rpm_to_force(self.rpm_spin[2])

         return left_force, right_force, bottom_force





    # -------------------------
    # Main
    # -------------------------
def main():
    xml_path = "radial_device.xml"  
    try:
        sim = TrajectoryMuJoCoIntegration(xml_path)
        print("MuJoCo model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
    num = int(input("Number of balls (default 1): ") or "1")
    interval = float(input("Interval between balls (default 2.0): ") or "2.0")
    sim.run_robot_drill(num_balls=num, ball_interval=interval)
    print("\n--- PASSIVE VIEWER MODE ---")
    sim.run_passive_simulation()
    print("Simulation completed.")
if __name__ == "__main__":
    main()