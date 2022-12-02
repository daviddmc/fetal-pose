import numpy as np
import os
import scipy.io as sio
from skimage.feature import peak_local_max
from pgmpy.models import MarkovModel
from pgmpy.inference import BeliefPropagation
from pgmpy.factors.discrete import DiscreteFactor
import pandas as pd
from math import gamma
from utils import load_yaml

data_partition = load_yaml("data_partition.yml")
data_train = data_partition["data_train"]
data_test = data_partition["data_test"]
data_val = data_partition["data_val"]

kp = [
    "ankle_l",
    "ankle_r",
    "knee_l",
    "knee_r",
    "bladder",
    "elbow_l",
    "elbow_r",
    "eye_l",
    "eye_r",
    "hip_l",
    "hip_r",
    "shoulder_l",
    "shoulder_r",
    "wrist_l",
    "wrist_r",
]

bone = np.array(
    [
        [9, 10],
        [11, 12],
        [7, 8],
        [11, 5],
        [12, 6],
        [5, 13],
        [6, 14],
        [9, 2],
        [10, 3],
        [2, 0],
        [3, 1],
        [4, 10],
        [7, 11],
        [4, 11],
    ]
)
num_a = 3  # 7
num_b = 3
num_peak = [
    num_a,
    num_a,
    num_a,
    num_a,
    num_b,
    num_a,
    num_a,
    num_b,
    num_b,
    num_a,
    num_a,
    num_b,
    num_b,
    num_a,
    num_a,
]

mu = []
sigma = []
GA_dict = {}
beta = 2


def init_GA():

    info_ga = pd.read_excel("../info_saved.xlsx")[["fnames", "W", "D"]]
    for index, row in info_ga.iterrows():
        GA_dict[
            str(row["fnames"]).zfill(6) if type(row["fnames"]) is int else row["fnames"]
        ] = (row["W"] if row["W"] else 32) + row["D"] / 7.0
    res = []
    for ds in data_train:
        ga = GA_dict[ds]
        joint_coord = sio.loadmat(os.path.join("../label", ds + ".mat"))["joint_coord"]
        bone_len = joint_coord[:, :, bone[:, 0]] - joint_coord[:, :, bone[:, 1]]
        bone_len = np.sqrt(np.sum(bone_len**2, 1))
        res.append(bone_len[:, :] * L_GA(32) / L_GA(ga))
    res = np.concatenate(res)
    for i in range(bone.shape[0]):
        tmp = res[:, i]
        tmp = tmp[tmp < 51]
        mu.append(np.mean(tmp))
        sigma.append(np.sqrt(np.var(tmp) * gamma(1 / beta) / gamma(3 / beta)))


def L_GA(ga):
    return ga * 0.67924 + 0.86298


def MRF(volume, J, dn):

    if len(mu) == 0:
        init_GA()

    ga = GA_dict[data_test[dn]]

    MM = MarkovModel()
    MM.add_nodes_from(kp)
    locs = []
    locs_value = []
    for i in range(J):
        locs.append(
            peak_local_max(
                volume[:, :, :, i],
                min_distance=3,
                exclude_border=True,
                indices=True,
                num_peaks=num_peak[i],
            )
        )
        locs_value.append(
            volume[locs[i][:, 0], locs[i][:, 1], locs[i][:, 2], i]
            - np.amin(volume[:, :, :, i])
        )
        factor = DiscreteFactor(
            [kp[i]], cardinality=[locs[i].shape[0]], values=locs_value[i]
        )
        MM.add_factors(factor)
    fac = L_GA(32) / L_GA(ga)
    for i, b in enumerate(bone):
        MM.add_edge(kp[b[0]], kp[b[1]])
        dr = np.sqrt(
            np.sum(
                (locs[b[0]].reshape((-1, 1, 3)) - locs[b[1]].reshape((1, -1, 3))) ** 2,
                2,
            )
        )
        edgePot = np.exp(-(((dr * fac - mu[i]) / sigma[i]) ** beta))
        edgePot[dr > 40] = 0
        factor = DiscreteFactor(
            [kp[b[0]], kp[b[1]]],
            cardinality=[locs[b[0]].shape[0], locs[b[1]].shape[0]],
            values=edgePot,
        )
        MM.add_factors(factor)
    # inference
    inference_method = BeliefPropagation(MM)
    config_max = inference_method.map_query()

    inds = []
    for i in range(J):
        inds.append(locs[i][config_max[kp[i]], :])
    return inds
