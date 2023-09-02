import torch
import EVFA
import GPG
from utils.cutom_env import *
from trainer import GeneralizedAC
from tqdm import tqdm

if __name__ == '__main__':
    random.seed(808)
    np.random.seed(808)
    torch.manual_seed(808)
    torch.cuda.manual_seed(808)

    map1 = MapInfo("maps/Weekday_Peak_network.csv")
    # map1 = MapInfo("maps/sioux_network.csv")
    env1 = Env(map1, 1, 87)

    # OP-GAC
    T = env1.LET_cost[0] * 1.025
    # GPG.WITH_WIS = True
    # EVFA.WITH_VARCON = True
    gac = GeneralizedAC(env1, time_budget=T, buffer_size=100, mode='on-policy', with_critic=True, device='cpu')
    # gac.supervised_warm_start(10000)
    gac.warm_start(60000, batch_size=100, epsilon=0)
    print(gac.eval(1, 1000))
    pi_score = gac.train(num_train=100, batch_size=100, with_eval=False, int_eval=1)
    print(gac.eval(1, 1000))

    # # GE-GAC + WIS + CV
    # T = 39
    # GPG.WITH_WIS = True
    # EVFA.WITH_VARCON = True
    # gac = GeneralizedAC(env1, time_budget=T, buffer_size=10000, mode='off-policy', with_critic=False, device='cpu')
    # gac.warm_start(100, epsilon=0.2)
    # pi_score = gac.train(num_train=100, batch_size=100, with_eval=False, int_eval=1)
    # print(gac.eval(1, 1000))

    # # LET-AC
    # T = 39
    # ac = GeneralizedAC(env1, time_budget=T, buffer_size=100, mode='let-ac', with_critic=False, device='cpu')
    # ac.warm_start(100, epsilon=0.2)
    # pi_score = ac.train(num_train=100, batch_size=100, with_eval=False, int_eval=1)
    # print(ac.eval(1, 1000))
