from typing import Callable

import gymnasium as gym
import torch
from gymnasium.vector import AutoresetMode


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    gamma: float = 0.99,
):
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, 0, capture_video, run_name, gamma)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos and "episode" in infos["final_info"]:
            fi = infos["final_info"]
            for r in fi["episode"]["r"][fi["_episode"]]:
                print(f"eval_episode={len(episodic_returns)}, episodic_return={r}")
                episodic_returns.append(r)
        obs = next_obs

    return episodic_returns


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    from algos.ppo_continuous_action import Agent, make_env

    model_path = hf_hub_download(
        repo_id="sdpkjc/Hopper-v4-ppo_continuous_action-seed1", filename="ppo_continuous_action.cleanrl_model"
    )
    evaluate(
        model_path,
        make_env,
        "Hopper-v4",
        eval_episodes=10,
        run_name=f"eval",
        Model=Agent,
        device="cpu",
        capture_video=False,
    )
