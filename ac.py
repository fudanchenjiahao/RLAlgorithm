import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym


# Actor-Critic网络
class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):                               ##服用前两层，增加稳定性
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actor_output = F.softmax(self.actor(x), dim=-1)
        critic_output = self.critic(x)
        return actor_output, critic_output


# AC算法
class AC:
    def __init__(self, input_shape, n_actions, gamma=0.99, lr=0.0005):
        self.actor_critic = ActorCritic(input_shape, n_actions)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma

    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        action = torch.LongTensor([action])
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([int(done)])

        # 计算Q值
        _, next_state_value = self.actor_critic(next_state)
        _, state_value = self.actor_critic(state)
        q_value = reward + self.gamma * next_state_value * (1 - done)    ##计算Q

        # 计算actor和critic的loss
        log_prob, _ = self.actor_critic(state)
        actor_loss = -(log_prob[0][action] * q_value).mean()
        critic_loss = F.mse_loss(state_value, q_value.detach())       #拟合V，用TD
        loss = actor_loss + critic_loss

        # 更新actor和critic的参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs, _ = self.actor_critic(state)
        action = probs.multinomial(num_samples=1)
        return action.item()



env = gym.make('CartPole-v1')
n_actions = env.action_space.n
state = env.reset()
input_shape = state.shape[0]
ac = AC(input_shape, n_actions)

for i in range(1000):                  ##1000个episode
    state = env.reset()
    done = False
    total_reward = 0

    while not done:                    ##一个episode内部，我们可以看到，只采样一条就更新了
        action = ac.get_action(state)
        next_state, reward, done, info = env.step(action)
        ac.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    print(f"Episode {i}: total reward = {total_reward}")
