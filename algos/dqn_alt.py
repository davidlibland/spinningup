"""Vanilla DQN implementation."""

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    capture_test_video: bool = False
    """whether to record a deterministic test episode at end of training (videos/{run_name}-test)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    gamma: float = 0.99
    """the discount factor gamma"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    learning_starts: int = 10000
    """timestep to start learning"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class QNet(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.qnet = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.SiLU(),
            layer_init(nn.Linear(64, 64)),
            nn.SiLU(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def forward(self, x):
        return self.qnet(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.num_iterations = args.total_timesteps // args.num_envs
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ],
        # Note, we use NEXT_STEP autoreset_mode (the gymnasium default).
        # See https://farama.org/Vector-Autoreset-Mode for details
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), (
        "only discrete action space is supported"
    )

    qnet = QNet(envs).to(device)
    optimizer = optim.Adam(qnet.parameters(), lr=args.learning_rate, eps=1e-5)
    target_network = QNet(envs).to(device)
    target_network.load_state_dict(qnet.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
        n_envs=args.num_envs,
    )

    global_step = 0
    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)
    obs = torch.Tensor(obs).to(device)

    for update in range(args.num_iterations):
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            int(args.exploration_fraction * args.num_iterations),
            update,
        )
        global_step += args.num_envs

        # Take an episilon-greedy action and update the replay buffer with the transition
        if random.random() < epsilon:
            action = torch.tensor(
                np.array(
                    [envs.single_action_space.sample() for _ in range(args.num_envs)]
                )
            ).to(device)
        else:
            with torch.no_grad():
                obs_t = torch.Tensor(obs).to(device)
                values = qnet(obs_t)
                action = values.argmax(dim=1)
        next_obs, reward, done, truncated, info = envs.step(action.cpu().numpy())
        rb.add(
            obs=obs,
            next_obs=next_obs,
            action=action,
            reward=reward,
            done=done,
            infos=info,
        )
        obs = next_obs
        if update < args.learning_starts:
            continue

        # NEXT_STEP autoreset mode: we reset the environment on the next step whenever
        # an episode ends (either terminated or truncated).
        # Record the epsodic return at the end of the episode (when done or truncated is True).
        if "episode" in info:
            episode_returns = info["episode"]["r"]
            episode_lengths = info["episode"]["l"]
            mask = info["_episode"]
            for ret, length in zip(episode_returns[mask], episode_lengths[mask]):
                print(f"global_step={global_step}, episodic_return={ret:.2f}")
                writer.add_scalar("charts/episodic_return", ret, global_step)
                writer.add_scalar("charts/episodic_length", length, global_step)

        # Sample a batch of experiences from the replay buffer
        b_obs, b_actions, b_next_obs, b_dones, b_rewards = rb.sample(args.batch_size)
        # Compute the target Q values
        with torch.no_grad():
            target_q_values = target_network(b_next_obs).max(dim=1, keepdim=True)[
                0
            ]  # (bs, 1)
            target_q_values = b_rewards + args.gamma * target_q_values * (
                1 - b_dones
            )  # (bs, 1)

        # Compute the current Q values
        q_values = qnet(b_obs).gather(1, b_actions.long())  # (bs, 1)
        # Compute the loss
        loss = nn.functional.mse_loss(q_values, target_q_values)

        # Take a gradient step
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(qnet.parameters(), args.max_grad_norm)
        optimizer.step()

        if update % args.target_network_frequency == 0:
            target_network.load_state_dict(qnet.state_dict())

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/qnet_loss", loss.item(), global_step)
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    if args.capture_test_video:
        test_env = gym.make(args.env_id, render_mode="rgb_array")
        test_env = gym.wrappers.RecordVideo(
            test_env, f"videos/{run_name}-test", episode_trigger=lambda _: True
        )
        test_env = gym.wrappers.RecordEpisodeStatistics(test_env)
        obs_t, _ = test_env.reset(seed=args.seed + 1000)
        qnet.eval()
        done = False
        while not done:
            with torch.no_grad():
                values = qnet(torch.Tensor(obs_t).unsqueeze(0).to(device))
                action_t = int(values.argmax(dim=-1).item())
            obs_t, _, terminated, truncated, info = test_env.step(action_t)
            done = bool(terminated or truncated)
        if "episode" in info:
            test_r = float(info["episode"]["r"])
            print(f"test_episodic_return={test_r}")
            writer.add_scalar("eval/test_episodic_return", test_r, global_step)
        test_env.close()

    envs.close()
    writer.close()
