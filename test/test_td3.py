import gym
import torch
import matplotlib.pyplot as plt
from agent.rl_utils import train_off_policy_agent, moving_average, ReplayBuffer
from agent.TD3 import TD3

# 超参数调整
actor_lr = 1e-4
critic_lr = 1e-3
num_episodes = 200
hidden_dim = 128
gamma = 0.98
tau = 0.005
policy_noise = 0.2
noise_clip = 0.5
policy_freq = 2
sigma_init = 0.1
buffer_size = 10000

# 环境初始化
env_name = 'Pendulum-v1'
env = gym.make(env_name, render_mode='human')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = TD3(
    state_dim=state_dim,
    hidden_dim=hidden_dim,
    action_dim=action_dim,
    action_bound=action_bound,
    sigma_init=sigma_init,
    actor_lr=actor_lr,
    critic_lr=critic_lr,
    tau=tau,
    gamma=gamma,
    device=device,
    policy_noise=policy_noise,
    noise_clip=noise_clip,
    policy_freq=policy_freq,
    num_episodes=num_episodes  # 新增参数
)

replay_buffer = ReplayBuffer(buffer_size)
return_list = train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size=1000, batch_size=64)

# 结果可视化
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('TD3 on {}'.format(env_name))
plt.show()

mv_return = moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('TD3 on {}'.format(env_name))
plt.show()