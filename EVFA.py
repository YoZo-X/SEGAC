import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from utils.networks import Probability
from utils.replay_buffer import TRANSITION, EpisodeReplayMemory, EpisodeBuffer
from utils.cutom_env import *
from utils.misc import soft_update

WITH_VARCON = False


class ExtendedVFA(nn.Module):
    def __init__(self, n_node, time_budget, learning_rate=1e-2, device='cpu'):
        super(ExtendedVFA, self).__init__()
        self.time_budget = time_budget
        self.device = device
        self.n_node = n_node
        self.model = Probability(state_size=n_node * 2 + 1, layer_size=128).to(device)
        self.target_model = Probability(state_size=n_node * 2 + 1, layer_size=128).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def learn(self, episodes, **kwargs):
        bp_loss = []
        for episode, travel_time in episodes:
            seq_len = len(episode)
            batch = episode.sample()
            state_tensor = torch.FloatTensor(np.array(batch.state)).view(seq_len, -1).to(self.device)
            next_state_tensor = torch.FloatTensor(np.array(batch.next_state)).view(seq_len, -1).to(self.device)
            done = list(batch.done)

            val = self.model(state_tensor)
            next_val = self.target_model(next_state_tensor)
            if done[-1] == 1 and travel_time < self.time_budget:
                next_val[-1] = 1

            bp_loss.append(F.mse_loss(val, next_val.detach()).sum().view(1, -1))

        bp_loss = torch.concat(bp_loss).mean()
        self.optimizer.zero_grad()
        bp_loss.backward()
        self.optimizer.step()
        soft_update(self.target_model, self.model, 0.1)

        return bp_loss.item()


class OffExtendedVFA(ExtendedVFA):
    def learn(self, episodes, **kwargs):
        cur_log_probs = kwargs['cur_log_probs']
        bp_loss = []
        for (episode, travel_time), cur_log_prob in zip(episodes, cur_log_probs):
            seq_len = len(episode)
            batch = episode.sample()

            state_tensor = torch.FloatTensor(np.array(batch.state)).view(seq_len, -1).to(self.device)
            next_state_tensor = torch.FloatTensor(np.array(batch.next_state)).view(seq_len, -1).to(self.device)
            done = list(batch.done)

            log_prob_tensor = torch.clone(torch.cat(batch.prob).view(seq_len, -1)).to(self.device)
            rau = torch.exp(cur_log_prob) / torch.exp(log_prob_tensor)
            # rau = torch.cumprod(rau, dim=0)

            val = self.model(state_tensor)
            next_val = self.model(next_state_tensor)
            if done[-1] == 1 and travel_time < self.time_budget:
                next_val[-1] = 1

            if WITH_VARCON:
                target = rau * next_val + (1 - rau) * val
            else:
                target = (next_val * rau)
            loss = F.mse_loss(val, target.detach()).sum()
            bp_loss.append(loss.view(1, -1))

        bp_loss = torch.concat(bp_loss)
        prios = bp_loss + 1e-8
        bp_loss = bp_loss.mean()
        self.optimizer.zero_grad()
        bp_loss.backward()
        self.optimizer.step()

        return bp_loss.item(), prios.data.cpu().numpy()

