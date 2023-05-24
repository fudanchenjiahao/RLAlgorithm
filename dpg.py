import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import random # 导入随机模块，用于采样
import numpy as np


# 定义一个策略网络，它输出给定状态的动作
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # 假设动作空间是-1到1之间
        return x


# 定义一个值函数网络，它输出给定状态和动作的Q值
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)  # 拼接状态和动作
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义一些超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 选择使用GPU或CPU
env_name = "Pendulum-v1"  # 选择一个连续控制的环境
gamma = 0.99  # 折扣因子
lr = 0.001  # 学习率
batch_size = 64  # 批量大小
max_episodes = 1000  # 最大训练回合数
max_steps = 200  # 每个回合的最大步数
exploration_noise = 0.1  # 探索噪声的标准差

# 创建环境和网络
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
policy_net = PolicyNetwork(state_dim, action_dim).to(device)  # 策略网络
value_net = ValueNetwork(state_dim, action_dim).to(device)  # 值函数网络
policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)  # 策略优化器
value_optimizer = torch.optim.Adam(value_net.parameters(), lr=lr)  # 值函数优化器

# 开始训练

for i_episode in range(max_episodes):
    transitions = []  # 创建一个空列表，用于存储转移
    state = env.reset()  # 重置环境，得到初始状态
    episode_reward = 0  # 记录每个回合的累积奖励
    for t in range(max_steps):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)  # 将状态转换为张量
        action_tensor = policy_net(state_tensor)  # 用策略网络输出动作
        action = action_tensor.cpu().detach().numpy()  # 将动作转换为numpy数组

        # 添加一些探索噪声，增加探索性
        noise = np.random.normal(0, exploration_noise, size=action_dim)
        action = action + noise

        # 在环境中执行动作，得到下一个状态，奖励和是否终止的标志
        next_state, reward, done, _ = env.step(action)

        # 更新累积奖励和状态
        episode_reward += reward
        state = next_state

        # 如果回合结束，跳出循环
        if done:
            break

        # 将当前的转移存储在一个列表中，用于后续的更新
        transition = (state, action, reward, next_state, done)
        transitions.append(transition)

    # 如果转移列表的长度大于批量大小，就进行一次更新
    if len(transitions) >= batch_size:
        # 从转移列表中随机采样一个批量
        batch = random.sample(transitions, batch_size)

        # 将批量中的数据分别转换为张量
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.as_tensor(states, dtype=torch.float32, device=device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(-1)

        # 计算目标Q值，使用下一个状态的最大Q值乘以折扣因子，再加上当前的奖励
        # 如果是终止状态，就只用当前的奖励
        with torch.no_grad():
            next_actions = policy_net(next_states)  # 用策略网络输出下一个动作
            next_q_values = value_net(next_states, next_actions)  # 用值函数网络输出下一个Q值
            target_q_values = rewards + (1 - dones) * gamma * next_q_values  # 计算目标Q值

        # 计算当前的Q值，使用当前的状态和动作
        current_q_values = value_net(states, actions)

        # 计算值函数网络的均方误差损失
        value_loss = F.mse_loss(current_q_values, target_q_values)

        # 优化值函数网络的参数
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # 计算策略网络的损失，使用负的Q值作为目标
        policy_loss = -value_net(states, policy_net(states)).mean()

        # 优化策略网络的参数
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

    # 打印每个回合的累积奖励
    print(f"Episode {i_episode}, Reward {episode_reward}")