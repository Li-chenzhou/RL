import numpy as np
import torch
import torch.nn.functional as F
from agent.AgentBase import AgentBase


class Qnet(torch.nn.Module):
    """只有一层隐藏层的Q网络"""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class VAnet(torch.nn.Module):
    '''只有一层隐藏层的A网络和V网络'''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet,self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  #共享网络部分
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        Q = V + A -A.mean(1).view(-1,1)      #Q值由V值和A值计算而得
        return Q

class DQN(AgentBase):
    """继承自AgentBase的DQN实现"""
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        # 初始化基类参数
        super().__init__(state_dim, hidden_dim, action_dim, device,
                         learning_rate=learning_rate, gamma=gamma,
                         # 通过kwargs传递额外参数
                         target_update=target_update,
                         epsilon=epsilon)

        # 初始化DQN特有参数
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0

        # 构建网络
        self.policy_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # 优化器
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),lr=self.learning_rate)

    def _init_additional_params(self, **kwargs):
        """初始化DQN特有参数"""
        self.target_update = kwargs.get('target_update', 10)
        self.epsilon = kwargs.get('epsilon', 0.01)

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = self._preprocess_state(state)
            action = self.policy_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']),dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        max_next_q_values = self.target_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        loss = torch.mean(F.mse_loss(q_values, q_targets))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.count += 1


    def max_q_value(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        return self.policy_net(state).max().item()


class DoubleDQN(DQN):
    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']),dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']),dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']),dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']),dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值

        max_action = self.policy_net(next_states).max(1)[1].view(-1, 1)
        max_next_q_values = self.target_net(next_states).gather(1, max_action)

        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())  # 更新目标网络
        self.count += 1


class DuelingDQN(DQN):
    """Dueling DQN实现，继承自DQN"""
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        # 初始化基类参数
        super().__init__(state_dim, hidden_dim, action_dim, learning_rate, gamma,
                         epsilon, target_update, device)

        # 重写网络构建部分，使用VAnet代替Qnet
        self.policy_net = VAnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_net = VAnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # 重新初始化优化器，因为网络结构已改变
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=self.learning_rate)