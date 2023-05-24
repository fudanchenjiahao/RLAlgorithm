import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import multiprocessing as mp
import numpy as np
import ipdb


# Actor-Critic网络
class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actor_output = F.softmax(self.actor(x), dim=-1)
        critic_output = self.critic(x)
        return actor_output, critic_output


# A3C更新函数
def update(global_model, optimizer, input_shape, n_actions, state, action, reward, next_state, done, gamma=0.99):

    state = torch.FloatTensor(state).unsqueeze(0)
    next_state = torch.FloatTensor(next_state).unsqueeze(0)
    action = torch.LongTensor([action])
    reward = torch.FloatTensor([reward])
    done = torch.FloatTensor([int(done)])

    # 计算advantage
    _, next_state_value = global_model(next_state)
    _, state_value = global_model(state)
    advantage = reward + gamma * next_state_value * (1 - done) - state_value

    # 计算actor和critic的loss
  #  print(state.shape)
    log_prob, _ = global_model(state)
    #print(log_prob.shape)
    actor_loss = -(log_prob[0][action] * advantage).mean()
    critic_loss = advantage.pow(2).mean()
    loss = actor_loss + critic_loss

    # 更新全局模型的参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 将全局模型的参数复制回本地模型
    local_model = ActorCritic(input_shape, n_actions)
    local_model.load_state_dict(global_model.state_dict())


# A3C Agent
def worker(global_model, optimizer, input_shape, n_actions, i):
    # 创建一个新的环境实例
    env = gym.make('CartPole-v1')

    # 创建一个新的本地模型，其初始参数与全局模型相同
    local_model = ActorCritic(input_shape, n_actions)
    local_model.load_state_dict(global_model.state_dict())

    # 对每一个Episode进行迭代
    for i_episode in range(1000):
        # 初始化环境并获取初始状态
        state = env.reset()
        # 初始化done标志位和总奖励
        done = False
        total_reward = 0

        # 当Episode未结束时进行迭代
        while not done:
            # 将state转换为Tensor以供模型使用
            state = torch.FloatTensor(state)
            # 使用本地模型得到动作概率和状态值
            probs, _ = local_model(state)
            # 根据动作概率选择动作
            action = probs.multinomial(num_samples=1).item()
            # 执行选择的动作并获取下一个状态、奖励和done标志位
            next_state, reward, done, info = env.step(action)
            # 更新总奖励
            total_reward += reward

            # 如果Episode已结束，则将next_state设置为全零
            if done:
                next_state = np.zeros(state.shape)
            else:
                next_state = np.array(next_state)

            # 异步更新全局模型
            update(global_model, optimizer, input_shape, n_actions, state.numpy(), action, reward, next_state, done)

            # 更新当前状态为next_state
            state = next_state

            # 如果Episode已结束，则跳出循环
            if done:
                break

        # 打印每个Episode的信息
        print(f"Worker {i}, Episode: {i_episode}, total reward: {total_reward}")




if __name__ == '__main__':
    # 创建环境。
    env = gym.make('CartPole-v1')

    # 获取环境中的动作数量。在CartPole-v1环境中，有两个可用的动作：向左移动和向右移动。
    n_actions = env.action_space.n

    # 获取状态的维度。在CartPole-v1环境中，每个状态是一个4维向量，代表杆子的位置、速度、角度和角速度。
    input_shape = env.observation_space.shape[0]

    # 创建全局模型。这是一个Actor-Critic模型，包括一个用于选择动作的Actor部分和一个用于评估状态价值的Critic部分。
    # 这个模型将在所有的工作进程之间共享。
    global_model = ActorCritic(input_shape, n_actions)

    # 允许全局模型在多进程中共享其内存。这是必要的，因为我们需要在所有的工作进程之间共享模型参数。
    global_model.share_memory()

    # 创建优化器。这个优化器将用于更新全局模型的参数。
    optimizer = optim.Adam(global_model.parameters(), lr=0.0005)

    # 创建一个空的进程列表。我们将在这个列表中存储所有的工作进程。
    processes = []

    # 创建和CPU核心数一样多的工作进程。每个工作进程都将运行worker函数，使用自己的环境副本进行探索，
    # 并异步地更新全局模型的参数。
    for i in range(mp.cpu_count()):
        p = mp.Process(target=worker, args=(global_model, optimizer, input_shape, n_actions, i))
        p.start()  # 启动进程
        processes.append(p)  # 将进程添加到进程列表中

    # 等待所有的工作进程完成。这是必要的，因为主进程需要等待所有的工作进程都完成他们的工作，
    # 才能安全地终止。如果主进程在工作进程还在运行时终止，那么这些工作进程可能会成为孤儿进程，继续无休止地运行。
    for p in processes:
        p.join()
