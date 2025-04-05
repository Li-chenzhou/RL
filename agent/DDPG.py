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


class DDPG(AgentBase):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, num_episodes,
                 sigma_init, actor_lr, critic_lr, tau, gamma, device):
        # 基类参数：critic_lr映射为learning_rate，其他通过kwargs传递
        super().__init__(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            device=device,
            learning_rate=critic_lr,  # 基类learning_rate对应critic_lr
            gamma=gamma,
            # 通过kwargs传递额外参数
            num_episodes=num_episodes,
            actor_lr=actor_lr,
            tau=tau,
            sigma_init=sigma_init,
            action_bound=action_bound
        )

        # 初始化策略网络和价值网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 优化器：critic使用基类learning_rate，actor使用子类actor_lr
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

    def _init_additional_params(self, **kwargs):
        """从kwargs中提取DDPG特有参数"""
        self.actor_lr = kwargs.get('actor_lr', 1e-3)
        self.tau = kwargs.get('tau', 0.01)
        self.sigma_init = kwargs.get('sigma_init', 0.1)
        self.action_bound = kwargs.get('action_bound', 1.0)
        self.num_episodes = kwargs.get('num_episodes', 200)
        self.sigma = self.sigma_init  #初始化动态噪声

    def take_action(self, state, episode=None):
        """使用基类预处理状态，并添加动态噪声"""
        state = self._preprocess_state(state)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()

        # 动态噪声衰减
        if episode is not None:
            self.sigma = max(self.sigma_init * (1 - episode / self.num_episodes), 0.01)

        noise = self.sigma * np.random.randn(self.action_dim)
        action = np.clip(action + noise, -self.action_bound, self.action_bound)
        return action

    def update(self, transition_dict):
        """更新逻辑，使用基类软更新方法"""
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).view(-1, self.action_dim).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)

        # Reward归一化
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Critic更新
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_q = self.target_critic(next_states, next_actions)
            q_targets = rewards + self.gamma * next_q * (1 - dones)

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)  # 梯度裁剪
        self.critic_optimizer.step()

        # Actor更新
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        # 使用基类软更新方法更新目标网络
        self._soft_update(self.target_actor, self.actor, self.tau)
        self._soft_update(self.target_critic, self.critic, self.tau)