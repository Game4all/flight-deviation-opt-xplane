import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO


class CriticComparisonWrapper(gym.Wrapper):
    """
    Un wrapper autour d'un environnement qui compare les gains de temps / distance entre suivre le cap magnétique du FMS et le gains de temps / distance de la déviation que propose le modèle.

    Q(s, a) = R(s, a) + gamma * V(s')
    """

    def __init__(self, env, model: PPO, gamma: float = 0.99):
        super().__init__(env)
        self.model = model
        self.gamma = gamma

        # Stats
        self.ppo_chosen_count = 0
        self.fms_chosen_count = 0
        self.total_steps = 0

    def step(self, action):
        # Evaluer l'action proposée par le modèle
        ppo_action = action
        ppo_next_obs, ppo_reward, ppo_done, ppo_info = self.env.evaluate_action(
            ppo_action)

        # Evaluer l'action de base du FMS
        fms_action = np.array([0.0, -1.0], dtype=np.float32)
        fms_next_obs, fms_reward, fms_done, fms_info = self.env.evaluate_action(
            fms_action)

        # 3. Get Values from Critic
        with torch.no_grad():
            obs_ppo_tensor = torch.as_tensor(
                ppo_next_obs).unsqueeze(0).to(self.model.device)
            obs_fms_tensor = torch.as_tensor(
                fms_next_obs).unsqueeze(0).to(self.model.device)

            v_ppo = self.model.policy.predict_values(obs_ppo_tensor).item()
            v_fms = self.model.policy.predict_values(obs_fms_tensor).item()

        # calculer la q-value de l'état avec l'équation de bellman Q(s, a) = R(s,a) + gamma * V(s')
        q_ppo = ppo_reward + (0 if ppo_done else self.gamma * v_ppo)
        q_fms = fms_reward + (0 if fms_done else self.gamma * v_fms)

        # prendre l'action avec la meilleure q-value (meilleur résultat)
        if q_ppo >= q_fms:
            chosen_action = ppo_action
            self.ppo_chosen_count += 1
            source = "PPO"
        else:
            chosen_action = fms_action
            self.fms_chosen_count += 1
            source = "FMS"

        self.total_steps += 1

        # on step le vrai environement
        obs, reward, terminated, truncated, info = self.env.step(chosen_action)

        # Update info with comparison data
        info["chosen_action_source"] = source
        info["q_ppo"] = q_ppo
        info["q_fms"] = q_fms
        info["v_ppo"] = v_ppo
        info["v_fms"] = v_fms
        info["ppo_reward_hyp"] = ppo_reward
        info["fms_reward_hyp"] = fms_reward

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.ppo_chosen_count = 0
        self.fms_chosen_count = 0
        self.total_steps = 0
        return self.env.reset(**kwargs)
