import torch
import torch.nn.functional as F
from agent.AgentBase import AgentBase


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class REINFORCE(AgentBase):
    '''基于蒙特卡洛估计的策略梯度算法'''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device):
        # 初始化基类参数
        super().__init__(state_dim, hidden_dim, action_dim, device,
                         learning_rate=learning_rate, gamma=gamma)
        # 策略网络
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 优化器
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def take_action(self, state):
        state = self._preprocess_state(state)  # 使用基类预处理状态
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        rewards = transition_dict['rewards']
        states = transition_dict['states']
        actions = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(rewards))):  # 从最后一步开始计算梯度
            reward = rewards[i]
            state = self._preprocess_state(states[i])  # 预处理状态
            action = torch.tensor([actions[i]], dtype=torch.long).view(-1, 1).to(self.device)

            # 计算对数概率
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G  # 累计损失
            loss.backward()  # 梯度累积
        self.optimizer.step()  # 统一更新参数