"""Vanilla implementation of policy gradient method."""

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
from einops import rearrange
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


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
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 1024
    """the number of steps to run in each environment per policy rollout"""
    gamma: float = 0.99
    """the discount factor gamma"""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


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


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_action(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_iterations = args.total_timesteps // args.batch_size
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

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.actor.parameters(), lr=args.learning_rate, eps=1e-5)

    # Setup storage for the training loop:
    observations = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    episode_start = np.zeros(envs.num_envs, dtype=bool)

    for update in range(args.num_iterations):
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            observations[step] = next_obs
            with torch.no_grad():
                action, *_ = agent.get_action(next_obs)
            next_obs, reward, done, truncated, info = envs.step(action.cpu().numpy())
            next_obs = torch.Tensor(next_obs).to(device)
            actions[step] = action
            rewards[step] = torch.Tensor(reward).to(device)
            episode_start = np.logical_or(done, truncated)
            dones[step] = torch.Tensor(episode_start).to(device)

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

        # Compute returns:
        returns = torch.zeros_like(rewards).to(device)
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                returns[t] = rewards[t]
            else:
                returns[t] = rewards[t] + args.gamma * returns[t + 1] * (1 - dones[t])

        # Flatten the batches:
        b_obs = rearrange(observations, "t n ... -> (t n) ...")
        b_actions = rearrange(actions, "t n ... -> (t n) ...")
        b_returns = rearrange(returns, "t n -> (t n)")

        # Losses:
        _, logprobs, entropies = agent.get_action(b_obs, b_actions.long())
        policy_loss = -(logprobs * b_returns).mean()
        entropy_loss = entropies.mean()

        loss = policy_loss - args.ent_coef * entropy_loss

        # Take gradient step:
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
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
        agent.eval()
        done = False
        while not done:
            with torch.no_grad():
                logits = agent.actor(torch.Tensor(obs_t).unsqueeze(0).to(device))
                action_t = int(torch.argmax(logits, dim=-1).item())
            obs_t, _, terminated, truncated, info = test_env.step(action_t)
            done = bool(terminated or truncated)
        if "episode" in info:
            test_r = float(info["episode"]["r"])
            print(f"test_episodic_return={test_r}")
            writer.add_scalar("eval/test_episodic_return", test_r, global_step)
        test_env.close()

    envs.close()
    writer.close()
