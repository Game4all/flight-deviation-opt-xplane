"""
Script pour calibrer les seuils pour le wrapper critique.
"""

import numpy as np
import argparse
from rich.console import Console
from rich.table import Table
from rich import box
from tqdm.rich import tqdm
from stable_baselines3 import PPO
from xp_sim_gym import OpenAPNavEnv, CriticComparisonWrapper, EnvironmentConfig, PlaneConfig, BenchmarkRouteGenerator


def run_simulation(env, model=None):
    """Simulates one route and returns stats."""
    obs, info = env.reset()
    done = False

    total_fuel = 0.0
    total_distance = 0.0

    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.array([0.0, -1.0], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_fuel += info.get("fuel_consumed", 0.0)
        total_distance += info.get("distance_flown", 0.0)

    return {
        "fuel_per_nm": total_fuel / (total_distance + 1e-6)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Grid search for optimal CriticComparisonWrapper threshold.")
    parser.add_argument("--min-threshold", type=float,
                        default=0.0, help="Minimum threshold value")
    parser.add_argument("--max-threshold", type=float,
                        default=0.5, help="Maximum threshold value")
    parser.add_argument("--steps", type=int, default=11,
                        help="Number of steps in the grid search")
    parser.add_argument("--gammas", type=float, nargs="+",
                        default=[0.92, 0.99], help="Gamma values to compare")
    parser.add_argument("--runs", type=int, default=250,
                        help="Number of routes to test per threshold")
    args = parser.parse_args()

    console = Console()
    console.rule("[bold cyan]Threshold & Gamma Grid Search[/bold cyan]")

    # 1. Config & Model
    plane_config = PlaneConfig(
        aircraft_type="A320", initial_lat=48.8566, initial_lon=2.3522)
    env_config = EnvironmentConfig()
    model_path = "ppo_flight_deviation_pretrained.zip"

    try:
        model = PPO.load(model_path)
        model.policy.set_training_mode(False)
        console.print(f"Loaded PPO model from {model_path}")
    except Exception as e:
        console.print(f"[bold red]Error loading model: {e}[/bold red]")
        return

    # 2. Pre-generate routes
    console.print(f"Generating {args.runs} benchmark routes...")
    generator = BenchmarkRouteGenerator(plane_config)
    routes_data = [generator.generate() for _ in range(args.runs)]

    # 3. Baseline calculation (FMS only)
    console.print("Calculating FMS Baseline...")
    fms_fuel_rates = []
    for route, wind in tqdm(routes_data, desc="Baseline"):
        env = OpenAPNavEnv(plane_config, env_config)
        env.set_nominal_route(route)
        env.set_wind_config(wind)
        stats = run_simulation(env, model=None)
        fms_fuel_rates.append(stats["fuel_per_nm"])

    console.print(
        f"Baseline Avg Fuel/NM: [bold]{np.mean(fms_fuel_rates):.4f}[/bold]\n")

    # 4. Grid Search
    thresholds = np.linspace(
        args.min_threshold, args.max_threshold, args.steps)
    results = []

    for gamma in args.gammas:
        console.print(f"\n[bold magenta]Testing Gamma: {gamma}[/bold magenta]")
        for thresh in thresholds:
            thresh = round(float(thresh), 4)
            console.print(f"  Threshold: [yellow]{thresh}[/yellow]")
            config_improvements = []

            for i, (route, wind) in enumerate(tqdm(routes_data, desc=f"G={gamma}, T={thresh}")):
                env = OpenAPNavEnv(plane_config, env_config)
                env.set_nominal_route(route)
                env.set_wind_config(wind)
                # Wrap with the current threshold and gamma
                wrapped_env = CriticComparisonWrapper(
                    env, model, threshold=thresh, gamma=gamma)

                stats = run_simulation(wrapped_env, model=model)
                run_rate = stats["fuel_per_nm"]
                baseline_rate = fms_fuel_rates[i]

                # Improvement for this specific run
                imp = (baseline_rate - run_rate) / (baseline_rate + 1e-9)
                config_improvements.append(imp)

            mean_imp = np.mean(config_improvements)
            worst_case_imp = np.min(config_improvements)

            results.append({
                "gamma": gamma,
                "threshold": thresh,
                "mean_improvement": mean_imp * 100,
                "worst_case_improvement": worst_case_imp * 100
            })

    # 5. Display Results
    table = Table(title="Threshold & Gamma Search (Risk-Based)",
                  box=box.ROUNDED)
    table.add_column("Gamma", justify="center")
    table.add_column("Threshold", justify="center")
    table.add_column("Mean Gain (%)", justify="right")
    table.add_column("Worst-Case Gain (%)", justify="right")

    best_config = None
    max_worst_case = -float('inf')

    for r in results:
        mean_c = "green" if r['mean_improvement'] > 0 else "red"
        worst_c = "green" if r['worst_case_improvement'] >= 0 else "red"

        # We optimize for the "least bad" worst case (max the min)
        if r['worst_case_improvement'] > max_worst_case:
            max_worst_case = r['worst_case_improvement']
            best_config = (r['gamma'], r['threshold'], r['mean_improvement'])

        table.add_row(
            str(r['gamma']),
            str(r['threshold']),
            f"[{mean_c}]{r['mean_improvement']:.2f}%[/{mean_c}]",
            f"[{worst_c}]{r['worst_case_improvement']:.2f}%[/{worst_c}]"
        )

    console.print(table)
    console.print(
        f"\n[bold green]Optimal risk-based configuration found: Gamma={best_config[0]}, Threshold={best_config[1]}[/bold green]")
    console.print(
        f"[bold green]Worst-case Gain: {max_worst_case:.2f}%, Mean Gain: {best_config[2]:.2f}%[/bold green]")


if __name__ == "__main__":
    main()
