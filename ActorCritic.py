import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from GPG import GeneralizedPG
from utils.misc import CategoricalMasked
from utils.networks import Policy
from utils.replay_buffer import TRANSITION, EpisodeReplayMemory, EpisodeBuffer
from utils.cutom_env import *

WITH_WIS = False


class ActorCritic(GeneralizedPG):
    def learn(self, episodes):
        policy_loss = []
        for episode, travel_time in episodes:
            seq_len = len(episode)
            batch = episode.sample()
            log_prob_tensor = torch.clone(torch.cat(batch.prob).view(seq_len, -1)).to(self.device)
            returns = np.ascontiguousarray(np.flip(np.array(batch.travel_time)))
            returns_tensor = torch.FloatTensor(returns).view(seq_len, -1).to(self.device)
            critic = np.ascontiguousarray(np.flip(np.array(batch.done)))
            critic_tensor = torch.FloatTensor(critic).view(seq_len, -1).to(self.device)

            policy_loss.append((-log_prob_tensor * (critic_tensor - returns_tensor).detach()).sum().view(1, -1))

        self.optimizer.zero_grad()
        policy_loss = torch.concat(policy_loss).mean()
        policy_loss.backward()
        self.optimizer.step()
        return policy_loss.item()


# def main():
#     map1 = MapInfo("maps/sioux_network.csv")
#     env = Env(map1, 1, 15)
#     env.reset()
#
#     ac = ActorCritic(n_node=env.map_info.n_node, time_budget=40, learning_rate=0.01, device='cpu')
#     buffer = EpisodeReplayMemory(100)
#
#     for i_episode in range(100):
#         score = 0
#         for _ in range(100):
#             epi_buf = EpisodeBuffer()
#             env.reset()
#             state = env.get_agent_obs_onehot() + [ac.time_budget - env.cost_time]
#             while True:
#                 state_tensor = torch.FloatTensor(state).to(ac.device)
#                 mask = env.get_agent_mask()
#                 mask_tensor = torch.FloatTensor(mask).to(ac.device)
#                 action, log_prob = ac.select_action(state_tensor.unsqueeze(0), mask_tensor)
#                 _, cost, done = env.step(action)
#                 next_state = env.get_agent_obs_onehot() + [ac.time_budget - env.cost_time]
#                 LET_cost = env.LET_cost[env.position - 1]
#                 epi_buf.push(state, action, next_state, env.cost_time, mask, log_prob, LET_cost)
#                 state = next_state
#                 if done or len(env.path) > env.map_info.n_node:
#                     score += 1 if env.cost_time < ac.time_budget else 0
#                     break
#             buffer.push(epi_buf, env.cost_time)
#         print('episodes:{}\tscore: {}'.format(i_episode + 1, score / 1000), env.path)
#         samples = buffer.sample(100)
#         ac.learn(samples)
#
#
# if __name__ == '__main__':
#     main()
