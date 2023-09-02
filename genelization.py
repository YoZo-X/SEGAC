import pandas as pd
import matplotlib.pyplot as plot
import torch
import EVFA
import GPG
from utils.cutom_env import *
from trainer import GeneralizedAC
from tqdm import tqdm

pd_raw = pd.read_excel('results/SiouxFalls_OD.xlsx')
map1 = MapInfo("maps/SiouxFalls_network.csv")
ODs = list(zip(pd_raw['O'].values, pd_raw['D'].values))

for OD in ODs:
    O = OD[0]
    D = OD[1]
    env1 = Env(map1, O, D)
    T = env1.LET_cost[O - 1] * 1
    gpg = GeneralizedAC(env1, time_budget=T, buffer_size=100, mode='on-policy', with_critic=False, device='cpu')
    epi_score = []
    for i_epi in range(1000):
        print(i_epi, end=':\t')
        gpg.load('results/Sioux_results/OD={}-{}/{}-{}_warm_episode={}.pth'.format(O, D, O, D, i_epi))
        epi_score.append(gpg.eval(1, 100))
