import numpy as np
import torch
import torch.nn.functional as F
from agent.AgentBase import AgentBase

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) * self.action_bound


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc_out(x)


class TD3(AgentBase):
    '''Twin Delayed Deep Deterministic Policy Gradient'''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 sigma_init, actor_lr, critic_lr, tau, gamma, device,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, num_episodes=200):
        # 基类参数：critic_lr映射为learning_rate，其他通过kwargs传递
        super().__init__(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            device=device,
            learning_rate=critic_lr,  # 基类learning_rate对应critic_lr
            gamma=gamma,
            # 通过kwargs传递额外参数
            actor_lr=actor_lr,
            tau=tau,
            sigma_init=sigma_init,
            action_bound=action_bound,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            policy_freq=policy_freq,
            num_episodes=num_episodes
        )

        # 初始化演员和评论家网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # 优化器：critic使用基类learning_rate，actor使用子类actor_lr
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=self.learning_rate)

        # 初始化其他参数
        self.total_it = 0  # 更新计数器

    def _init_additional_params(self, **kwargs):
        """从kwargs中提取TD3特有参数"""
        self.actor_lr = kwargs.get('actor_lr', 1e-3)
        self.tau = kwargs.get('tau', 0.005)
        self.sigma_init = kwargs.get('sigma_init', 0.1)
        self.action_bound = kwargs.get('action_bound', 1.0)
        self.num_episodes = kwargs.get('num_episodes', 200)
        self.policy_noise = kwargs.get('policy_noise', 0.2)
        self.noise_clip = kwargs.get('noise_clip', 0.5)
        self.policy_freq = kwargs.get('policy_freq', 2)
        self.sigma = self.sigma_init  # 初始化动态噪声

    def take_action(self, state, episode=None):
        """使用基类预处理状态，并添加动态噪声"""
        state = self._preprocess_state(state)  # 调用基类方法处理状态
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()

        # 动态噪声衰减（基于episode）
        if episode is not None:
            self.sigma = max(self.sigma_init * (1 - episode / self.num_episodes), 0.01)

        noise = self.sigma * np.random.randn(self.action_dim)
        return np.clip(action + noise, -self.action_bound, self.action_bound)

    def update(self, transition_dict):
        """覆盖基类update方法，实现TD3更新逻辑"""
        self.total_it += 1
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).view(-1, self.action_dim).to(
            self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)

        with torch.no_grad():
            # 生成目标动作并添加噪声
            next_actions = self.target_actor(next_states)
            noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(-self.action_bound, self.action_bound)

            # 双重目标Q值
            target_Q1 = self.target_critic_1(next_states, next_actions)
            target_Q2 = self.target_critic_2(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            q_targets = rewards + self.gamma * target_Q * (1 - dones)

        # 更新双重评论家
        current_Q1 = self.critic_1(states, actions)
        current_Q2 = self.critic_2(states, actions)
        critic_loss = F.mse_loss(current_Q1, q_targets) + F.mse_loss(current_Q2, q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_optimizer.param_groups[0]['params'], 1.0)
        self.critic_optimizer.step()

        # 延迟策略更新
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic_1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 使用基类软更新方法更新目标网络
            self._soft_update(self.target_actor, self.actor, self.tau)
            self._soft_update(self.target_critic_1, self.critic_1, self.tau)
            self._soft_update(self.target_critic_2, self.critic_2, self.tau)