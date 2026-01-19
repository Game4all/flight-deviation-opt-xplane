import math
import gymnasium as gym
import time
import numpy as np
import pygame
from xp_sim_gym.openap_env import OpenAPNavEnv
from xp_sim_gym.viz_wrapper import OpenAPVizWrapper
from xp_sim_gym.config import PlaneEnvironmentConfig

def generate_long_route(lat_0, lon_0, num_segments=8):
    route = [{'lat': lat_0, 'lon': lon_0, 'alt': 10000}]
    curr_lat, curr_lon = lat_0, lon_0
    curr_bng = 45 # Northeast
    
    for _ in range(num_segments):
        dist_nm = np.random.uniform(30, 60)
        turn = np.random.uniform(-45, 45) # Sharper turns to test auto-pilot
        curr_bng = (curr_bng + turn) % 360
        
        d_lat = dist_nm / 60.0 * math.cos(math.radians(curr_bng))
        d_lon = dist_nm / 60.0 * math.sin(math.radians(curr_bng)) / math.cos(math.radians(curr_lat))
        
        curr_lat += d_lat
        curr_lon += d_lon
        route.append({'lat': curr_lat, 'lon': curr_lon, 'alt': 10000})
    return route

def main():
    print("Initializing Autopilot Visualization...")
    
    # 1. Setup config (Stage 2: Moderate Wind to show it works with wind)
    lat_start, lon_start = 48.8566, 2.3522 # Paris
    long_route = generate_long_route(lat_start, lon_start, num_segments=32)

    config = PlaneEnvironmentConfig(
        aircraft_type="A320",
        initial_lat=lat_start,
        initial_lon=lon_start,
        nominal_route=long_route,
        randomize_wind=True # Enable wind to verify crab angle logic
    )
    
    # 2. Create env and wrap it
    env = OpenAPNavEnv(config)
    env.set_pretraining_stage(3) # Moderate wind
    
    try:
        env = OpenAPVizWrapper(env, width=1024, height=768)
        print("Visualization Window Created.")
    except Exception as e:
        print(f"Error creating visualization (headless environment?): {e}")
        return

    print(f"Starting Autopilot Test with {len(long_route)} waypoints.")
    print("Action is FIXED at [0, 0] (Autonomous Mode).")
    
    obs, info = env.reset()
    
    done = False
    step_count = 0
    
    max_xte = 0.0
    
    try:
        running = True
        while running and not done:
            # FORCE ZERO ACTION (Autonomous Mode)
            # The agent sends [0, 0], meaning "Follow the computed course"
            action = np.array([0.0, 0.0], dtype=np.float32)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Extract metrics
            current_xte = env.env._calculate_xte() 
            max_xte = max(max_xte, abs(current_xte))
            
            # Render
            env.render()
            
            # Interactive check for quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Delay for viewing comfort (Step represents ~5 mins, so this is fast-forward)
            # But with sub-stepping, the trajectory will look smooth relative to waypoints.
            time.sleep(0.5) 
            
            step_count += 1
            if step_count % 5 == 0:
                print(f"Step {step_count}: XTE={current_xte:.4f} NM | Wind=({env.env.wind_u:.1f}, {env.env.wind_v:.1f})")

        print("\nSimulation Finished.")
        print(f"Max XTE observed: {max_xte:.4f} NM")
        
        # Hold window open for a moment
        time.sleep(2)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        env.close()
        pygame.quit()

if __name__ == "__main__":
    main()
