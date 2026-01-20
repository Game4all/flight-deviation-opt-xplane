import math
import gymnasium as gym
import time
import numpy as np
from xp_sim_gym.openap_env import OpenAPNavEnv
from xp_sim_gym.viz_wrapper import OpenAPVizWrapper
from xp_sim_gym.config import PlaneEnvironmentConfig

from stable_baselines3 import PPO

from xp_sim_gym.route_generator import RouteStageGenerator

def main():
    # 1. Setup config (Stage 5 for complex route + wind)
    lat_start, lon_start = 48.8566, 2.3522
    
    base_config = PlaneEnvironmentConfig(
        aircraft_type="A320",
        initial_lat=lat_start,
        initial_lon=lon_start,
    )
    generator = RouteStageGenerator(base_config)
    route, wind_streams = generator.generate(stage=5)

    config = PlaneEnvironmentConfig(
        aircraft_type="A320",
        initial_lat=route[0]['lat'],
        initial_lon=route[0]['lon'],
        nominal_route=route,
        wind_streams=wind_streams,
        randomize_wind=False
    )
    
    # 2. Create env and wrap it
    env = OpenAPNavEnv(config)
    env.set_pretraining_stage(5)
    env = OpenAPVizWrapper(env)
    
    # 3. Load Model
    model_path = "ppo_flight_deviation_pretrained.zip"
    try:
        model = PPO.load(model_path, env=env)
        print(f"Loaded pretrained model from {model_path}")
        use_model = True
    except Exception as e:
        print(f"Could not load model: {e}. Falling back to random actions.")
        use_model = False

    print(f"Starting simulation test with {len(route)} waypoints...")
    obs, info = env.reset()
    
    done = False
    step_count = 0
    
    try:
        while not done and step_count < 300:
            # 4. Get action from model or random
            if use_model:
                action, _states = model.predict(obs, deterministic=True)
                print(action)
            else:
                action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 5. Render
            env.render()
            
            # Slow down for visibility
            time.sleep(0.5) # Faster now that it's a model
            step_count += 1
            
            if step_count % 10 == 0:
                print(f"Step {step_count}, Reward: {reward:.2f}, XTE: {env.env._calculate_xte():.2f}")
                
        print("Simulation finished.")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
