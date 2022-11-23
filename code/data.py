import tensorflow as tf
import scipy.io as sio
import os
import nibabel as nib
import numpy as np
from random import shuffle, randint, choice, uniform, random
from math import ceil
from dataset import get_dataset_from_indexable
from scipy.ndimage import zoom

data_train = [
    "010918L",
    "010918S",
    "013018S",
    "013118S",
    "021218S",
    "022318L",
    "022318S",
    "031317L",
    "031317T",
    "031615",
    "031616",
    "032217",
    "032318b",
    "032318c",
    "032318d",
    "032818",
    "040218",
    "040417",
    "041017",
    "041318L",
    "043015",
    "050318L",
    "051718L",
    "051718S",
    "051817",
    "052218L",
    "052218S",
    "052418L",
    "053017",
    "061217",
    "062817S",
    "071717S",
    "072017S",
    "080217",
    "082117S",
    "082517L",
    "082917b",
    "091917S",
    "092117L",
    "092117S",
    "092817L",
    "100317L",
    "102617",
    "103017a",
    "103017b",
    "110217L",
    "111017L",
    "111017S",
    "121517b",
]
# data_train = ['022618', '031615', '031616', '032318a', '040218','040716', '043015', '061217', '102617']
data_val = [
    "013018L",
    "013118L",
    "032318a",
    "040716",
    "052418S",
    "053117L",
    "062117",
    "062817L",
    "071218",
    "071717L",
    "082117L",
    "083017S",
    "101317",
    "121517a",
]
data_test = [
    "021218L",
    "022618",
    "041818",
    "052516",
    "053117S",
    "072017L",
    "082917a",
    "083017L",
    "083115",
    "090517L",
    "091917L",
    "100317S",
    "110214",
    "120717",
]

train_data_list_all = []
val_data_list_all = []
test_data_list_all = []
train_data_list = []
val_data_list = []
test_data_list = []
train_k = 0
val_k = 0
test_k = 0

preload_data = False

rots = [
    [],
    [(1, (0, 1))],
    [(1, (1, 2))],
    [(1, (2, 0))],
    [(2, (0, 1))],
    [(1, (0, 1)), (1, (1, 2))],
    [(1, (0, 1)), (1, (2, 0))],
    [(1, (1, 2)), (1, (0, 1))],
    [(2, (1, 2))],
    [(1, (2, 0)), (1, (1, 2))],
    [(2, (2, 0))],
    [(3, (0, 1))],
    [(2, (0, 1)), (1, (1, 2))],
    [(2, (0, 1)), (1, (2, 0))],
    [(2, (1, 2)), (1, (2, 0))],
    [(1, (0, 1)), (2, (1, 2))],
    [(1, (0, 1)), (2, (2, 0))],
    [(1, (1, 2)), (2, (0, 1))],
    [(3, (1, 2))],
    [(3, (2, 0))],
    [(3, (0, 1)), (1, (1, 2))],
    [(2, (0, 1)), (1, (1, 2)), (1, (0, 1))],
    [(3, (1, 2)), (1, (2, 0))],
    [(1, (0, 1)), (3, (1, 2))],
]


def vec_rot(loc, k, axes):
    loc1, loc0, loc2 = loc
    loc_ = [loc0, loc1, loc2]
    for _ in range(k):
        loc_[axes[1]], loc_[axes[0]] = loc_[axes[0]], loc_[axes[1]]
        loc_[axes[0]] = -loc_[axes[0]]
    return np.stack((loc_[1], loc_[0], loc_[2]))


def vec_flip(loc, axis):
    if axis == 0:  # y
        loc[1] = -loc[1]
    elif axis == 1:  # x
        loc[0] = -loc[0]
    else:  # Z
        loc[2] = -loc[2]
    return loc


def random_rot(v, h, loc):
    rot = choice(rots)
    for k, axes in rot:
        v = np.rot90(v, k, axes)
        h = np.rot90(h, k, axes)
        loc = vec_rot(loc, k, axes)
    return v, h, loc


def random_flip(v, h, loc):
    flip = choice([None, 0, 1, 2])
    flip_order1 = [1, 0, 3, 2, 4, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13]
    if flip is not None:
        v = np.flip(v, flip)
        h = np.flip(h, flip)
        h = h[:, :, :, flip_order1]
        loc = vec_flip(loc, flip)
        loc = loc[:, flip_order1]
    return v, h, loc


def read_nifti(nii_filename):
    data = nib.load(nii_filename)
    return np.squeeze(data.get_data().astype(np.float32))


class Dataset:
    def __init__(self, opts):
        record = sio.loadmat(opts.record_path)["record"][0]
        rec_dict = {}
        for name, n in zip(record["name"], record["n"]):
            rec_dict[name[0]] = n
        train_dict = []
        val_dict = []
        test_dict = []

        if opts.train_all:
            data_train.extend(data_val)
            data_train.extend(data_test)

        for folder in [
            os.path.join(opts.rawdata_path, f)
            for f in sorted(os.listdir(opts.rawdata_path))
        ]:
            folder_basename = os.path.basename(folder)
            if folder_basename in rec_dict:
                # get joint coord label
                label_filename = os.path.join(opts.label_path, folder_basename + ".mat")

                if opts.downsample_label:
                    joint_coord = (
                        np.around(sio.loadmat(label_filename)["joint_coord"] / 2) * 2
                    )
                    joint_coord = joint_coord.astype(np.int32)
                else:
                    joint_coord = np.around(
                        sio.loadmat(label_filename)["joint_coord"]
                    ).astype(np.int32)

                # get filenames
                niinames = [
                    os.path.join(folder, f)
                    for f in sorted(os.listdir(folder))
                    if f.endswith(".nii.gz")
                ]

                d = {
                    "foldername": folder_basename,
                    "filenames": niinames,
                    "labels": joint_coord,
                    "n": rec_dict[folder_basename],
                }
                if folder_basename in data_train:
                    train_dict.append(d)
                elif folder_basename in data_val:
                    val_dict.append(d)
                else:
                    test_dict.append(d)
        self.dataset_dict = {"train": train_dict, "val": val_dict, "test": test_dict}

    def get_data_list(self, stage):
        datasets = self.dataset_dict[stage]
        data_list = []
        for dn, dataset in enumerate(datasets):
            n = np.nonzero(dataset["n"])[0]
            if stage == "train":
                if n.size < 10:
                    n = n.repeat(int(np.ceil(10 / n.size)))
            for i in n:
                d = {
                    "filename": dataset["filenames"][i],
                    "label": dataset["labels"][i],
                    "foldername": dn,
                }
                if preload_data:
                    d["volume"] = read_nifti(d["filename"])
                data_list.append(d)
        if stage == "train" or stage == "val":
            shuffle(data_list)
        return data_list

    def get_output_type_shape(self, stage, opts):
        if stage == "test":
            return (tf.float32, tf.int32, tf.int64, tf.int64), (
                [None, None, None, 1],
                [3, opts.nJoint],
                [3],
                [],
            )
        else:  # train
            return (tf.float32, tf.float32, tf.float32), (
                [None, None, None, 1],
                [None, None, None, opts.nJoint],
                [3, opts.nJoint],
            )

    def init_datalist(self, stage):
        data_list = self.get_data_list(stage)
        if stage == "train":
            global train_data_list_all
            global train_data_list
            train_data_list_all = data_list
            print("# of train data: %d" % len(train_data_list_all))
            train_data_list = train_data_list_all[: (40 * len(data_train))]
            # train_data_list = train_data_list_all
        elif stage == "val":
            global val_data_list_all
            global val_data_list
            val_data_list_all = data_list
            print("# of val data: %d" % len(val_data_list_all))
            val_data_list = val_data_list_all[: (20 * len(data_val))]
        else:
            global test_data_list_all
            global test_data_list
            test_data_list_all = data_list
            print("# of test data: %d" % len(test_data_list_all))
            test_data_list = test_data_list_all

    def _get_dataset(self, stage, opts):
        is_shuffle = stage == "train"
        batch_size = 1 if stage == "test" else opts.batch_size
        dtype, shape = self.get_output_type_shape(stage, opts)
        self.init_datalist(stage)
        if stage == "train":
            f = lambda i: train_map_fn(train_data_list[i], opts)
            L = len(train_data_list)
        elif stage == "val":
            f = lambda i: val_map_fn(val_data_list[i], opts)
            L = len(val_data_list)
        else:
            f = lambda i: test_map_fn(test_data_list[i], opts)
            L = len(test_data_list)
        # map_fn = getattr(sys.modules[__name__], stage + '_map_fn')
        # f = lambda i: map_fn(data_list[i], opts)
        return get_dataset_from_indexable(f, dtype, shape, L, batch_size, is_shuffle)

    def get_dataset(self, opts):
        print("construct dataset")
        if opts.run == "test":
            print("get dataset for testing")
            return self._get_dataset("test", opts)
        else:
            print("get dataset for training and validation")
            return self._get_dataset("train", opts), self._get_dataset("val", opts)

    def update_datalist(self, stage):
        if stage == "train":
            global train_k
            train_k = train_k + len(train_data_list)
            L = len(train_data_list_all)
            for i in range(len(train_data_list)):
                train_data_list[i] = train_data_list_all[(i + train_k) % L]
        elif stage == "val":
            global val_k
            val_k = val_k + len(val_data_list)
            L = len(val_data_list_all)
            for i in range(len(val_data_list)):
                val_data_list[i] = val_data_list_all[(i + val_k) % L]
        else:
            raise Exception("error")


def train_map_fn(data_dict, opts):
    volume, heatmap, loc = val_map_fn(data_dict, opts)
    if opts.rot:
        volume, heatmap, loc = random_rot(volume, heatmap, loc)
    if opts.flip:
        volume, heatmap, loc = random_flip(volume, heatmap, loc)
    # if opts.scale:
    #    volume[:, :, :, 0] = (
    #        uniform(1 - opts.scale, 1 + opts.scale) * volume[:, :, :, 0]
    #    )
    if opts.scale and opts.norm:
        volume[:, :, :, 0] = volume[:, :, :, 0] ** uniform(
            1 - opts.scale, 1 + opts.scale
        )
    return volume, heatmap, loc


def val_map_fn(data_dict, opts):
    if preload_data:
        volume = data_dict["volume"]
    else:
        volume = read_nifti(data_dict["filename"])
    joint_coord = data_dict["label"]
    if random() < opts.zoom:
        if opts.zoom <= 1:
            zf = uniform(1 / opts.zoom_factor, opts.zoom_factor)
        else:
            zf = opts.zoom_factor
        volume = zoom(volume, zf, order=1)
        joint_coord = np.around(joint_coord * zf).astype(np.int32)
    if opts.norm:
        percentile_fac = np.percentile(volume[volume > 0], 99)
        # mean = np.mean(volume)
        # std = np.std(volume)
    volume, heatmap, loc = crop_trainval(
        volume, joint_coord, opts.crop_size, opts.mag, opts.sigma
    )
    if opts.norm:
        volume = volume / percentile_fac
        # volume = (volume - mean) / std
    return volume, heatmap, loc


def test_map_fn(data_dict, opts):
    if preload_data:
        volume = data_dict["volume"]
    else:
        volume = read_nifti(data_dict["filename"])
    joint_coord = data_dict["label"]
    if opts.norm:
        volume = volume / np.percentile(volume[volume > 0], 99)
        # volume = (volume - np.mean(volume)) / np.std(volume)
    return (
        crop_test(volume, opts.test_arg),
        joint_coord,
        volume.shape,
        data_dict["foldername"],
    )


def crop_trainval(volume, joint_coord, crop_size, mag, sigma):
    # size crop size
    crop_size_x, crop_size_y, crop_size_z = crop_size
    # generate random point
    s = volume.shape
    if crop_size_x == 0:
        pad_width = [
            (
                int((ceil(ss / 8.0) * 8 - ss) / 2),
                int(ceil((ceil(ss / 8.0) * 8 - ss) / 2)),
            )
            for ss in volume.shape
        ]
        volume = np.pad(volume, pad_width, "reflect")  #'constant', 'edge'
        x_0, y_0, z_0 = -pad_width[1][0], -pad_width[0][0], -pad_width[2][0]
        crop_size_y, crop_size_x, crop_size_z = volume.shape
    elif s[2] >= crop_size_z:
        x_0, y_0, z_0 = (
            randint(0, s[1] - crop_size_x),
            randint(0, s[0] - crop_size_y),
            randint(0, s[2] - crop_size_z),
        )  # close interval
        volume = volume[
            y_0 : y_0 + crop_size_y, x_0 : x_0 + crop_size_x, z_0 : z_0 + crop_size_z
        ]
    else:
        x_0, y_0, z_0 = (
            randint(0, s[1] - crop_size_x),
            randint(0, s[0] - crop_size_y),
            0,
        )
        volume = volume[y_0 : y_0 + crop_size_y, x_0 : x_0 + crop_size_x, :]
        volume = np.concatenate(
            (
                volume,
                np.zeros((crop_size_y, crop_size_x, crop_size_z - s[2]), volume.dtype),
            ),
            axis=2,
        )

    # generate heatmap
    y_range = np.reshape(
        np.arange(y_0 + 1, y_0 + crop_size_y + 1, dtype=np.float32), (-1, 1, 1, 1)
    )
    x_range = np.reshape(
        np.arange(x_0 + 1, x_0 + crop_size_x + 1, dtype=np.float32), (1, -1, 1, 1)
    )
    z_range = np.reshape(
        np.arange(z_0 + 1, z_0 + crop_size_z + 1, dtype=np.float32), (1, 1, -1, 1)
    )

    def gen_hmap(joint):
        x_label, y_label, z_label = np.reshape(joint, (3, 1, 1, 1, -1))
        dx, dy, dz = x_range - x_label, y_range - y_label, z_range - z_label
        dd = dx**2 + dy**2 + dz**2
        if sigma:
            return (
                mag
                * ((2.0 / sigma) ** 3)
                * np.exp((-1.0 / 2.0 / sigma**2) * dd, dtype=np.float32)
            )
        else:
            return dd.astype(np.float32)

    heatmap = gen_hmap(joint_coord)

    loc = (
        2
        * (joint_coord.astype(np.float32) - [[x_0 + 1], [y_0 + 1], [z_0 + 1]])
        / [[crop_size_x - 1], [crop_size_y - 1], [crop_size_z - 1]]
        - 1.0
    )

    return np.expand_dims(volume, -1), heatmap, loc.astype(np.float32)


def crop_test(volume, test_arg):
    pad_width = [
        (int((ceil(s / 8.0) * 8 - s) / 2), int(ceil((ceil(s / 8.0) * 8 - s) / 2)))
        for s in volume.shape
    ]
    # pad_width = [(int((ceil(s/16.0)*16-s)/2), int(ceil((ceil(s/16.0)*16-s)/2))) for s in volume.shape]
    v = np.pad(volume, pad_width, "reflect")  #'constant', 'edge'
    if test_arg[0] is not None:
        v = np.rot90(v, 1, test_arg[0])
    if test_arg[1] is not None:
        v = np.flip(v, test_arg[1])
    v = np.expand_dims(v, -1)
    return v
