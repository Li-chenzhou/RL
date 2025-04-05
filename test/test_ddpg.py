import random
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from agent.rl_utils import train_off_policy_agent, moving_average, ReplayBuffer
from agent.DDPG import DDPG


actor_lr = 1e-4
critic_lr = 1e-3
num_episodes = 200
hidden_dim = 128
gamma = 0.98
tau = 0.01
buffer_size = 10000
minimal_size = 1000
batch_size = 64
sigma_init = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 环境初始化
env_name = 'Pendulum-v1'
env = gym.make(env_name,render_mode='human')
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# 修改训练循环以传递episode参数
replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, num_episodes,
             sigma_init, actor_lr, critic_lr, tau, gamma, device)

return_list = train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format(env_name))
plt.show()

mv_return = moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format(env_name))
plt.show()