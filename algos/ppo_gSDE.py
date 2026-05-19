# PPO for continuous control with generalized State-Dependent Exploration (gSDE).
# gSDE follows Raffin et al. 2021 / the Stable-Baselines3
# StateDependentNoiseDistribution: exploration noise is a linear map of the
# (detached) policy latent through a random matrix resampled every
# `sde_sample_freq` steps, so exploration is smooth in state and time.
# Base PPO derived from cleanrl ppo_continuous_action.py.
import math
import copy
import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from gymnasium.vector import AutoresetMode
from torch.distributions.normal import Normal
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
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_test_video: bool = False
    """whether to record a deterministic test episode at end of training (videos/{run_name}-test)"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    noise_halflife: float | None = None
    """The half life for autocorrelated noise"""
    use_sde: bool = False
    """if toggled, use generalized State-Dependent Exploration (gSDE) instead of
    i.i.d. (or AR-correlated) action noise"""
    sde_sample_freq: int = -1
    """resample the gSDE exploration matrix every this many env steps
    (<=0 means resample once per rollout, matching SB3's reset_noise cadence)"""
    sde_log_std_init: float = 0.0
    """initial value of the gSDE log-std parameter. Matches SB3's
    ActorCriticPolicy default (the SB3-zoo Pendulum config does not override
    it). Empirically, a smaller init (e.g. -2.0) starves exploration here and
    the policy collapses before it learns."""
    sde_learn_features: bool = False
    """gSDE: if False (SB3 default) the latent feeding the exploration
    noise/variance is detached; True couples it to the policy gradient
    (the original naive implementation)"""
    bootstrap_truncation: bool = True
    """if True (SB3-faithful) add gamma*V(terminal_obs) to the reward on
    TimeLimit truncation; if False the cleanrl behavior (truncation treated
    as terminal) is used"""
    normalize: bool = True
    """if toggled, apply the MuJoCo-style wrapper stack (NormalizeObservation
    + obs clip + NormalizeReward + reward clip). Default True preserves the
    cleanrl behavior; --no-normalize disables it (matches the SB3-zoo Pendulum
    setup, which runs with normalize=False)"""
    eval_episodes: int = 20
    """after training, evaluate the DETERMINISTIC policy (action = mean, no
    exploration noise) over this many episodes — the metric SB3 reports"""
    optuna_report_path: str = None
    """if set, append JSONL lines `{"global_step": s, "return": r}` once per
    iteration so an Optuna driver can score/prune the run"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma, normalize=True):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(
            env
        )  # deal with dm_control's Dict observation space
        # RecordEpisodeStatistics is kept inside any reward normalization so the
        # logged episodic_return is always the raw environment return.
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        if normalize:
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(
                env,
                lambda obs: np.clip(obs, -10, 10),
                observation_space=gym.spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=env.observation_space.shape,
                    dtype=env.observation_space.dtype,
                ),
            )
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(
                env, lambda reward: np.clip(reward, -10, 10)
            )
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# gSDE numerical floor on the induced variance (SB3's StateDependentNoise eps).
SDE_EPSILON = 1e-6


class Agent(nn.Module):
    def __init__(self, envs, noise_halflife=None, use_sde=False,
                 sde_log_std_init=0.0, learn_features=False):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # Split actor into a feature trunk + linear mean head so gSDE can use
        # the policy latent features. `actor_mean(x)` keeps the old API.
        self.actor_trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.actor_head = layer_init(nn.Linear(64, act_dim), std=0.01)
        # Used by the vanilla / AR(1) paths only.
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

        # AR(1)-correlated exploration noise (mutually exclusive with gSDE).
        if noise_halflife is not None and noise_halflife > 0:
            self.alpha = 0.5 ** (1 / noise_halflife)
        else:
            self.alpha = None

        # gSDE (Raffin et al. 2021 / SB3 StateDependentNoiseDistribution).
        self.use_sde = bool(use_sde)
        # SB3 default learn_features=False: features feeding the gSDE
        # noise/variance are detached so the trunk is trained only by the
        # mean's gradient. True couples them (the original naive bug).
        self.learn_features = bool(learn_features)
        self._n_features = 64
        self._act_dim = act_dim
        if self.use_sde:
            # full_std=True -> one log-std per (feature, action) pair.
            self.sde_log_std = nn.Parameter(
                torch.ones(self._n_features, act_dim) * sde_log_std_init
            )
            self.exploration_mat = None        # (n_features, act_dim)
            self.exploration_matrices = None   # (batch, n_features, act_dim)
            self.sample_sde_weights(1, self.sde_log_std.device)

    # --- gSDE helpers (mirrors SB3 StateDependentNoiseDistribution) ---
    def get_sde_std(self):
        # use_expln=False branch: std = exp(log_std).
        return torch.exp(self.sde_log_std)

    def sample_sde_weights(self, batch_size, device):
        """Resample the gSDE exploration matrices from N(0, std)."""
        std = self.get_sde_std()
        dist = Normal(torch.zeros_like(std), std)
        self.exploration_mat = dist.rsample().to(device)
        self.exploration_matrices = dist.rsample((batch_size,)).to(device)

    def _sde_noise(self, latent_sde):
        # SB3 get_noise: per-env exploration matrices when shapes line up,
        # otherwise fall back to the shared exploration matrix.
        if (
            self.exploration_matrices is None
            or len(latent_sde) == 1
            or len(latent_sde) != len(self.exploration_matrices)
        ):
            return latent_sde @ self.exploration_mat
        noise = torch.bmm(latent_sde.unsqueeze(1), self.exploration_matrices)
        return noise.squeeze(1)

    def actor_mean(self, x):
        return self.actor_head(self.actor_trunk(x))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, lastnoise=None):
        latent = self.actor_trunk(x)
        action_mean = self.actor_head(latent)

        if self.use_sde:
            # learn_features=False (SB3 default): detach so the trunk is
            # trained only by the mean; exploration adapts via sde_log_std.
            latent_sde = latent if self.learn_features else latent.detach()
            std = self.get_sde_std()
            variance = (latent_sde**2) @ (std**2)
            action_std = torch.sqrt(variance + SDE_EPSILON)
            probs = Normal(action_mean, action_std)
            if action is None:
                action = action_mean + self._sde_noise(latent_sde)
            return (
                action,
                probs.log_prob(action).sum(1),
                probs.entropy().sum(1),
                self.critic(x),
                None,
            )

        # Vanilla i.i.d. Gaussian, optionally AR(1)-correlated.
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        noise = None

        if action is None:
            if self.alpha:
                if lastnoise is None:
                    noise = torch.randn_like(action_mean)
                else:
                    noise = lastnoise * self.alpha + math.sqrt(
                        1 - self.alpha**2
                    ) * torch.randn_like(action_mean)
                action = action_mean + action_std * noise
            else:
                action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
            noise,
        )


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
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
            make_env(args.env_id, i, args.capture_video, run_name, args.gamma,
                     normalize=args.normalize)
            for i in range(args.num_envs)
        ],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    agent = Agent(
        envs,
        noise_halflife=args.noise_halflife,
        use_sde=args.use_sde,
        sde_log_std_init=args.sde_log_std_init,
        learn_features=args.sde_learn_features,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    noise = None

    recent_returns: deque[float] = deque(maxlen=100)
    optuna_fh = None
    if args.optuna_report_path:
        os.makedirs(os.path.dirname(args.optuna_report_path) or ".", exist_ok=True)
        optuna_fh = open(args.optuna_report_path, "a", buffering=1)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # gSDE: SB3 resets the exploration matrices at the start of every
        # rollout, and additionally every `sde_sample_freq` steps if > 0.
        if args.use_sde:
            agent.sample_sde_weights(args.num_envs, device)

        for step in range(0, args.num_steps):
            if (
                args.use_sde
                and args.sde_sample_freq > 0
                and step > 0
                and step % args.sde_sample_freq == 0
            ):
                agent.sample_sde_weights(args.num_envs, device)
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, noise = agent.get_action_and_value(
                    next_obs, lastnoise=noise
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)

            # SB3-faithful timeout bootstrapping: for envs that ended via
            # TimeLimit truncation (not true termination), add
            # gamma * V(terminal_obs) so a fixed-horizon cutoff is not
            # treated as the end of the world. Pendulum *always* truncates
            # at 200 steps, so without this every episode's value targets
            # are systematically biased.
            if args.bootstrap_truncation and "final_obs" in infos:
                boot = (
                    np.asarray(infos["_final_obs"], dtype=bool)
                    & np.asarray(truncations, dtype=bool)
                    & ~np.asarray(terminations, dtype=bool)
                )
                if boot.any():
                    idxs = np.nonzero(boot)[0]
                    term_obs = np.stack(
                        [np.asarray(infos["final_obs"][i], dtype=np.float32)
                         for i in idxs]
                    )
                    with torch.no_grad():
                        tv = agent.get_value(
                            torch.tensor(term_obs, device=device)
                        ).reshape(-1)
                    idx_t = torch.as_tensor(idxs, device=device)
                    rewards[step, idx_t] += args.gamma * tv

            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(next_done).to(device),
            )

            # With SAME_STEP autoreset, RecordEpisodeStatistics' output is nested
            # under `final_info` (since the returned obs is the reset state).
            if "final_info" in infos and "episode" in infos["final_info"]:
                fi = infos["final_info"]
                done_mask = fi["_episode"]
                for r, l in zip(
                    fi["episode"]["r"][done_mask], fi["episode"]["l"][done_mask]
                ):
                    print(f"global_step={global_step}, episodic_return={r:.2f}")
                    writer.add_scalar("charts/episodic_return", r, global_step)
                    writer.add_scalar("charts/episodic_length", l, global_step)
                    recent_returns.append(float(r))

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

        if optuna_fh is not None and recent_returns:
            optuna_fh.write(json.dumps({
                "global_step": int(global_step),
                "return": float(np.mean(recent_returns)),
                "n_recent": len(recent_returns),
            }) + "\n")

    if optuna_fh is not None:
        optuna_fh.close()

    # Deterministic evaluation — the metric SB3 reports (action = policy mean,
    # no exploration noise). Built with the same obs-side wrappers as training;
    # if normalizing, reuse the trained obs statistics (frozen).
    if args.eval_episodes > 0:
        eval_env = make_env(
            args.env_id, 0, False, f"{run_name}-eval", args.gamma,
            normalize=args.normalize,
        )()
        if args.normalize:
            eval_env.set_wrapper_attr(
                "obs_rms",
                copy.deepcopy(envs.envs[0].get_wrapper_attr("obs_rms")),
            )
            try:
                eval_env.set_wrapper_attr("update_running_mean", False)
            except (AttributeError, ValueError):
                pass
        agent.eval()
        eval_returns = []
        for ep in range(args.eval_episodes):
            obs_e, _ = eval_env.reset(seed=args.seed + 10_000 + ep)
            done = False
            while not done:
                with torch.no_grad():
                    a = agent.actor_mean(
                        torch.Tensor(obs_e).unsqueeze(0).to(device)
                    ).squeeze(0).cpu().numpy()
                obs_e, _, term_e, trunc_e, info_e = eval_env.step(a)
                done = bool(term_e or trunc_e)
            if "episode" in info_e:
                eval_returns.append(float(info_e["episode"]["r"]))
        eval_env.close()
        agent.train()
        if eval_returns:
            arr = np.asarray(eval_returns, dtype=np.float64)
            mean, std = float(arr.mean()), float(arr.std())
            print(
                f"DET_EVAL episodes={len(arr)} mean={mean:.2f} std={std:.2f} "
                f"min={arr.min():.2f} max={arr.max():.2f}"
            )
            writer.add_scalar("eval/det_return_mean", mean, global_step)
            writer.add_scalar("eval/det_return_std", std, global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(
                args,
                episodic_returns,
                repo_id,
                "PPO",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
            )

    if args.capture_test_video:
        # Rebuild the test env with the same observation-side wrappers as
        # training so the policy operates on inputs in the distribution it
        # was trained on. Reward-normalization wrappers are intentionally
        # omitted — RecordEpisodeStatistics then reports raw returns.
        test_env = gym.make(args.env_id, render_mode="rgb_array")
        test_env = gym.wrappers.RecordVideo(
            test_env, f"videos/{run_name}-test", episode_trigger=lambda _: True
        )
        test_env = gym.wrappers.FlattenObservation(test_env)
        test_env = gym.wrappers.RecordEpisodeStatistics(test_env)
        test_env = gym.wrappers.ClipAction(test_env)
        if args.normalize:
            test_env = gym.wrappers.NormalizeObservation(test_env)
            test_env = gym.wrappers.TransformObservation(
                test_env,
                lambda obs: np.clip(obs, -10, 10),
                observation_space=gym.spaces.Box(
                    low=-10.0, high=10.0,
                    shape=test_env.observation_space.shape,
                    dtype=test_env.observation_space.dtype,
                ),
            )
            # Copy obs running stats from a training sub-env so the test env
            # normalizes consistently, and freeze updates during evaluation.
            test_env.set_wrapper_attr(
                "obs_rms",
                copy.deepcopy(envs.envs[0].get_wrapper_attr("obs_rms")),
            )
            try:
                test_env.set_wrapper_attr("update_running_mean", False)
            except (AttributeError, ValueError):
                pass

        obs_t, _ = test_env.reset(seed=args.seed + 1000)
        agent.eval()
        done = False
        while not done:
            with torch.no_grad():
                # Deterministic policy: action mean (matches discrete ppo.py's
                # argmax behavior for evaluation).
                x = torch.Tensor(obs_t).unsqueeze(0).to(device)
                action = agent.actor_mean(x).squeeze(0).cpu().numpy()
            obs_t, _, terminated, truncated, info = test_env.step(action)
            done = bool(terminated or truncated)
        if "episode" in info:
            test_r = float(info["episode"]["r"])
            print(f"test_episodic_return={test_r}")
            writer.add_scalar("eval/test_episodic_return", test_r, global_step)
        test_env.close()

    envs.close()
    writer.close()
