import numpy as np
import os
from math import ceil
from mrf import MRF


def test_result(outputs, joint_coord, s, dn, opts):

    outputs = np.squeeze(outputs)
    outputs = outputs[:, :, :, : opts.nJoint]
    if opts.test_arg[1] is not None:
        flip_order = [1, 0, 3, 2, 4, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13]
        outputs = np.flip(outputs, opts.test_arg[1])
        outputs = outputs[:, :, :, flip_order]
    if opts.test_arg[0] is not None:
        outputs = np.rot90(outputs, -1, opts.test_arg[0])
    joint_coord = np.squeeze(joint_coord)
    s = np.squeeze(s)
    dn = np.squeeze(dn)

    predict_mean_coord = np.zeros_like(joint_coord) * 0.0

    pad_width = [
        (int((ceil(ss / 8.0) * 8 - ss) / 2), int(ceil((ceil(ss / 8.0) * 8 - ss) / 2)))
        for ss in s
    ]
    # pad_width = [(int((ceil(ss/16.0)*16-ss)/2), int(ceil((ceil(ss/16.0)*16-ss)/2))) for ss in s]
    sss = outputs.shape
    volume = outputs[
        pad_width[0][0] : sss[0] - pad_width[0][1],
        pad_width[1][0] : sss[1] - pad_width[1][1],
        pad_width[2][0] : sss[2] - pad_width[2][1],
    ]

    volume[volume < 0] = 0
    # xv, yv, zv = np.meshgrid(np.arange(1, s[1]+1), np.arange(1, s[0]+1), np.arange(1, s[2]+1))

    if opts.use_MRF:
        inds = MRF(volume, joint_coord.shape[-1], dn)

    for i in range(joint_coord.shape[-1]):
        if joint_coord[2, i] <= 0:
            joint_coord[:, i] = np.nan
            predict_mean_coord[:, i] = np.nan
        else:

            if opts.use_MRF:
                ind = inds[i]
            else:
                ind = np.unravel_index(np.argmax(volume[:, :, :, i]), s)

            x_avg, y_avg, z_avg = average_coord(volume[:, :, :, i], ind)
            predict_mean_coord[0, i] = x_avg
            predict_mean_coord[1, i] = y_avg
            predict_mean_coord[2, i] = z_avg

    return np.hstack((dn, predict_mean_coord.ravel(), joint_coord.ravel()))


def average_coord(volume, ind):
    weights = 0
    x_p = y_p = z_p = 0
    for x in range(ind[1] - 1, ind[1] + 2):
        for y in range(ind[0] - 1, ind[0] + 2):
            for z in range(ind[2] - 1, ind[2] + 2):
                if (
                    0 <= x < volume.shape[1]
                    and 0 <= y < volume.shape[0]
                    and 0 <= z < volume.shape[2]
                ):
                    weights += volume[y, x, z]
                    x_p += x * volume[y, x, z]
                    y_p += y * volume[y, x, z]
                    z_p += z * volume[y, x, z]
    return x_p / weights + 1, y_p / weights + 1, z_p / weights + 1


def save_test_result(res, opts):
    if res:
        headers = ["data_id"]
        for prefix in ["predict_", "label_"]:
            for suffix in ["_x", "_y", "_z"]:
                for j in opts.joint:
                    headers.append(prefix + str(j + 1) + suffix)
        np.savetxt(
            os.path.join(opts.output_path, opts.name, opts.name + ".csv"),
            np.vstack(res),
            fmt="%.3f",
            delimiter=",",
            header=",".join(headers),
            comments="",
        )
