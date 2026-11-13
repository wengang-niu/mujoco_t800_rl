"""05 — 手写 PPO（PyTorch，不用 stable-baselines3）。

自己实现共享主干的 Actor-Critic（高斯策略 + 价值网络）、GAE 优势估计和
clip 目标，训练 04 的 T800StandEnv 学会站立。

用法：
  python 05_ppo.py --smoke          # 先在 1 维简单环境自检 PPO 实现能学到
  python 05_ppo.py --train 2000     # 在 T800 上训练 2000 个 rollout 迭代
  python 05_ppo.py --eval path.pt   # 评估已保存模型
  python 05_ppo.py --train 1000 --gpu   # 强制用 GPU（见下方说明）

关于设备：本任务是「单环境 + 小网络」（观测 47 维、动作 12 维、隐藏层 128）。
实测 GPU 反而更慢——rollout 每步都要把 47 维观测搬进/搬出显存并同步，
2048 步的 act() 在 GPU 上 1.6s、CPU 上仅 0.75s；PPO 批量更新两者相当
（0.46s vs 0.38s）。因此默认用 CPU。只有将来改成「多环境并行 + 大批量」
时 GPU 才划算，届时加 --gpu 即可。
"""
from __future__ import annotations

import argparse
import importlib
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

DEVICE = torch.device("cpu")


# ---- 网络 ----

def mlp(dims: list[int]) -> nn.Sequential:
    """隐藏层用 Tanh，最后一层保持线性（价值头需要无界输出）。"""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """共享主干：一次前向同时给出动作均值与价值。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.trunk = mlp([obs_dim, hidden, hidden])
        self.mean_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))  # 初始 std≈0.37（平衡需精细控制）

    def _forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        mean = torch.tanh(self.mean_head(h))            # 均值限制在 (-1,1)
        value = self.value_head(h).squeeze(-1)
        return mean, value

    def std(self) -> torch.Tensor:
        return self.log_std.exp().clamp(min=1e-4)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self._forward(obs)
        dist = Normal(mean, self.std().expand_as(mean))
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), value

    def act_mean(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self._forward(obs)
        return mean

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        """一次前向返回 (新 logp, 熵, 价值)，供 PPO 更新用。"""
        mean, value = self._forward(obs)
        dist = Normal(mean, self.std().expand_as(mean))
        logp = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, entropy, value


# ---- 简易 1 维自检环境（验证 PPO 实现无误）----

class TinyEnv:
    """1 维点质量调节：动作推 x 回原点，奖励 -x^2。PPO 应能很快学到。"""

    def __init__(self):
        self.x = 0.0
        self.t = 0

    def reset(self):
        self.x = float(np.random.uniform(-1, 1))
        self.t = 0
        return np.array([self.x], dtype=np.float32), {}

    def step(self, a: np.ndarray):
        a = float(np.clip(np.asarray(a).ravel()[0], -1, 1))
        self.x = float(np.clip(self.x + 0.1 * a, -2, 2))
        self.t += 1
        done = self.t >= 100
        return np.array([self.x], dtype=np.float32), -self.x ** 2, done, done, {}


# ---- PPO 训练核心 ----

class PPO:
    def __init__(self, obs_dim: int, act_dim: int, *, lr=3e-4, gamma=0.99,
                 lam=0.95, clip=0.2, epochs=10, batch_size=256,
                 ent_coef=0.001, vf_coef=0.5, max_grad_norm=0.5, hidden=128):
        self.ac = ActorCritic(obs_dim, act_dim, hidden).to(DEVICE)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=lr)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.batch_size = epochs, batch_size
        self.ent_coef, self.vf_coef, self.max_grad_norm = ent_coef, vf_coef, max_grad_norm

    def train_iteration(self, rollout: dict) -> dict:
        """用一轮 rollout 数据做多 epoch 的 PPO 更新。"""
        obs = torch.as_tensor(rollout["obs"], dtype=torch.float32, device=DEVICE)
        act = torch.as_tensor(rollout["act"], dtype=torch.float32, device=DEVICE)
        old_logp = torch.as_tensor(rollout["logp"], dtype=torch.float32, device=DEVICE)
        adv = torch.as_tensor(rollout["adv"], dtype=torch.float32, device=DEVICE)
        ret = torch.as_tensor(rollout["ret"], dtype=torch.float32, device=DEVICE)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # 优势归一化

        n = len(obs)
        idx = np.arange(n)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                b = idx[start:start + self.batch_size]
                ob, ac, olp, ad, rt = obs[b], act[b], old_logp[b], adv[b], ret[b]
                new_logp, entropy, value = self.ac.evaluate(ob, ac)
                ratio = torch.exp(new_logp - olp)
                surr1 = ratio * ad
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * ad
                policy_loss = -torch.min(surr1, surr2).mean()
                ent = entropy.mean()
                value_loss = 0.5 * (value - rt).pow(2).mean()
                loss = policy_loss - self.ent_coef * ent + self.vf_coef * value_loss

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.opt.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += ent.item()
        k = n // self.batch_size * self.epochs
        return {key: val / max(k, 1) for key, val in stats.items()}

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> tuple[np.ndarray, float, float]:
        o = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
        action, logp, value = self.ac.act(o)
        return action.cpu().numpy(), float(logp.item()), float(value.item())

    @torch.no_grad()
    def act_deterministic(self, obs: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
        return self.ac.act_mean(o).cpu().numpy()

    @torch.no_grad()
    def act_batch(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """批量采样：obs [n,obs_dim] -> (action [n,act], logp [n], value [n])。"""
        o = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
        action, logp, value = self.ac.act(o)
        return action.numpy(), logp.numpy(), value.numpy()

    def save(self, path: str) -> None:
        torch.save(self.ac.state_dict(), path)

    def load(self, path: str) -> None:
        self.ac.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))


def collect_rollout(env, ppo: PPO, n_steps: int) -> tuple[dict, list, list]:
    """收集 n_steps 步数据并计算 GAE，返回 (rollout, 回合奖励, 回合长度)。"""
    obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
    rew_buf, done_buf = [], []
    obs, _ = env.reset()
    ep_rewards, ep_lens, ep_return, ep_len = [], [], 0.0, 0

    for _ in range(n_steps):
        action, logp, value = ppo.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        obs_buf.append(obs); act_buf.append(action); logp_buf.append(logp)
        val_buf.append(value); rew_buf.append(reward); done_buf.append(done)
        ep_return += reward; ep_len += 1
        obs = next_obs
        if done:
            ep_rewards.append(ep_return); ep_lens.append(ep_len)
            ep_return = 0.0; ep_len = 0
            obs, _ = env.reset()

    # GAE：从最后一步回推优势
    last_val = _bootstrap_value(ppo, obs, done_buf[-1])
    adv = np.zeros(n_steps)
    gae = 0.0
    for i in reversed(range(n_steps)):
        next_val = last_val if i == n_steps - 1 else val_buf[i + 1]
        delta = rew_buf[i] + ppo.gamma * next_val * (1 - done_buf[i]) - val_buf[i]
        gae = delta + ppo.gamma * ppo.lam * (1 - done_buf[i]) * gae
        adv[i] = gae
    ret = adv + np.array(val_buf)
    rollout = {"obs": np.array(obs_buf), "act": np.array(act_buf),
               "logp": np.array(logp_buf), "adv": adv, "ret": ret}
    return rollout, ep_rewards, ep_lens


def _bootstrap_value(ppo: PPO, obs: np.ndarray, done: bool) -> float:
    """终局引导值：若最后一步已结束则为 0，否则用价值网络估计。"""
    if done:
        return 0.0
    o = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
    _, value = ppo.ac._forward(o)
    return float(value.item())


def collect_rollout_vec(venv, ppo: PPO, n_steps: int) -> tuple[dict, list, list]:
    """多环境批量收集：venv.n 个环境各跑 n_steps//n 步，逐环境算 GAE。"""
    n = venv.n
    per = n_steps // n
    obs_buf = np.zeros((per, n, venv.obs.shape[-1]), dtype=np.float32)
    act_buf = np.zeros((per, n, ppo.ac.mean_head.out_features), dtype=np.float64)
    logp_buf = np.zeros((per, n), dtype=np.float64)
    val_buf = np.zeros((per, n), dtype=np.float64)
    rew_buf = np.zeros((per, n), dtype=np.float64)
    done_buf = np.zeros((per, n), dtype=bool)

    ep_ret = np.zeros(n); ep_len = np.zeros(n)
    ep_rewards, ep_lens = [], []

    obs = venv.obs
    for i in range(per):
        actions, logp, values = ppo.act_batch(obs)
        next_obs, rewards, dones = venv.step(actions)
        obs_buf[i] = obs; act_buf[i] = actions; logp_buf[i] = logp
        val_buf[i] = values; rew_buf[i] = rewards; done_buf[i] = dones
        ep_ret += rewards; ep_len += 1
        for j in range(n):
            if dones[j]:
                ep_rewards.append(ep_ret[j]); ep_lens.append(ep_len[j])
                ep_ret[j] = 0.0; ep_len[j] = 0
        obs = next_obs

    # 终局引导值（done 的为 0，其余用价值网络估计）
    o = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        _, v = ppo.ac._forward(o)
    last_val = np.where(done_buf[-1], 0.0, v.numpy())

    # 逐环境 GAE
    adv = np.zeros((per, n))
    gae = np.zeros(n)
    for i in reversed(range(per)):
        next_val = last_val if i == per - 1 else val_buf[i + 1]
        delta = rew_buf[i] + ppo.gamma * next_val * (1 - done_buf[i]) - val_buf[i]
        gae = delta + ppo.gamma * ppo.lam * (1 - done_buf[i]) * gae
        adv[i] = gae
    ret = adv + val_buf
    rollout = {"obs": obs_buf.reshape(-1, obs_buf.shape[-1]),
               "act": act_buf.reshape(-1, act_buf.shape[-1]),
               "logp": logp_buf.reshape(-1), "adv": adv.reshape(-1),
               "ret": ret.reshape(-1)}
    return rollout, ep_rewards, ep_lens


# ---- 主流程 ----

def make_stand_env():
    """按文件名导入 04 的环境（模块名以数字开头，需用 importlib）。"""
    return importlib.import_module("04_stand_env").T800StandEnv()


class VecStandEnv:
    """进程内多环境：n 个 T800 实例顺序步进，供批量采样摊薄网络开销。"""

    def __init__(self, n: int):
        self.envs = [make_stand_env() for _ in range(n)]
        self.n = n
        self.obs = np.stack([e.reset()[0] for e in self.envs]).astype(np.float32)

    def step(self, actions: np.ndarray):
        obs_list, rew = [], np.zeros(self.n)
        dones = np.zeros(self.n, dtype=bool)
        for i, e in enumerate(self.envs):
            o, r, term, trunc, _ = e.step(actions[i])
            d = term or trunc
            if d:
                o, _ = e.reset()
            obs_list.append(o); rew[i] = r; dones[i] = d
        self.obs = np.stack(obs_list).astype(np.float32)
        return self.obs, rew, dones


def train(collect, ppo: PPO, iters: int, n_steps: int, save_dir: str, log_every=20):
    os.makedirs(save_dir, exist_ok=True)
    print(f"device={DEVICE}  iters={iters}  n_steps={n_steps}", flush=True)
    for it in range(1, iters + 1):
        t0 = time.time()
        rollout, ep_rewards, ep_lens = collect(n_steps)
        stats = ppo.train_iteration(rollout)
        mean_ep = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        mean_len = float(np.mean(ep_lens)) if ep_lens else 0.0
        if it % log_every == 0:
            print(f"iter {it:5d} | mean_ep_return={mean_ep:8.1f} mean_len={mean_len:6.1f} | "
                  f"policy_loss={stats['policy_loss']:7.4f} value_loss={stats['value_loss']:7.4f} "
                  f"entropy={stats['entropy']:.4f} | {time.time()-t0:.1f}s", flush=True)
        if it % 200 == 0:
            ppo.save(os.path.join(save_dir, f"ppo_stand_{it}.pt"))
    ppo.save(os.path.join(save_dir, "ppo_stand_final.pt"))
    print("saved:", os.path.join(save_dir, "ppo_stand_final.pt"), flush=True)


def smoke_test() -> None:
    """在 1 维环境自检 PPO 能否学到（mean_ep_return 应从 ~-33 升到接近 0）。"""
    env = TinyEnv()
    ppo = PPO(1, 1)
    print("自检：1 维环境上训练 60 轮，观察 mean_ep_return（初始约 -33，应升到接近 0）")
    for it in range(1, 61):
        rollout, ep_rewards, _ = collect_rollout(env, ppo, 256)
        ppo.train_iteration(rollout)
        if it % 15 == 0:
            print(f"  iter {it:3d}: mean_ep_return={np.mean(ep_rewards):.4f}")
    final = np.mean(ep_rewards)
    print("自检" + ("通过 ✅" if final > -5.0 else "未通过 ❌"))


def eval_model(env, ppo: PPO, episodes: int = 5) -> None:
    returns, lens = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        ep_ret, ep_len = 0.0, 0
        for _ in range(env.max_steps):
            action = ppo.act_deterministic(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward; ep_len += 1
            if terminated or truncated:
                break
        returns.append(ep_ret); lens.append(ep_len)
    print(f"eval: mean_return={np.mean(returns):.1f}  mean_len={np.mean(lens):.1f} "
          f"(满分约 {1000 * 3.4:.0f})")


def main() -> None:
    global DEVICE
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="1 维环境自检")
    p.add_argument("--train", type=int, default=0, metavar="N", help="T800 训练迭代数")
    p.add_argument("--eval", type=str, default="", help="评估模型路径")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nenvs", type=int, default=8, help="并行环境数（批量采样摊薄网络开销）")
    p.add_argument("--gpu", action="store_true", help="强制用 GPU（默认 CPU，单环境小网络下 CPU 更快）")
    args = p.parse_args()

    if args.gpu and torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    if args.smoke:
        smoke_test()
        return

    if args.train:
        ppo = PPO(39, 4)
        if args.nenvs > 1:
            venv = VecStandEnv(args.nenvs)
            collect = lambda n: collect_rollout_vec(venv, ppo, n)
        else:
            env = make_stand_env()
            collect = lambda n: collect_rollout(env, ppo, n)
        n_steps = (2048 // args.nenvs) * args.nenvs  # 保证被环境数整除
        train(collect, ppo, args.train, n_steps=n_steps, save_dir="checkpoints")
        return

    if args.eval:
        env = make_stand_env()
        ppo = PPO(39, 4)
        ppo.load(args.eval)
        eval_model(env, ppo)
        return

    p.print_help()


if __name__ == "__main__":
    main()
