import torch
import gym
import matplotlib.pyplot as plt
from agent.rl_utils import train_on_policy_agent, moving_average
from agent.TRPO import TRPO

num_episodes = 1000
hidden_dim = 128
gamma = 0.99
lmbda = 0.95
critic_lr = 1e-2
kl_constraint = 0.001
alpha = 0.5
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'CartPole-v1'
env = gym.make(env_name,render_mode='human')

torch.manual_seed(0)
agent = TRPO(hidden_dim, env.observation_space, env.action_space, lmbda,kl_constraint, alpha, critic_lr, gamma, device)
return_list = train_on_policy_agent(env, agent, num_episodes)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('TRPO on {}'.format(env_name))
plt.show()

mv_return = moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('TRPO on {}'.format(env_name))
plt.show()
