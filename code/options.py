import argparse
import os
import datetime
from utils import *


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.initialized = False

    def initialize(self):

        # path
        self.parser.add_argument(
            "--rawdata_path", default="../newdata/", type=str, help="raw data path"
        )
        self.parser.add_argument(
            "--label_path", default="../label/", type=str, help="label path"
        )
        self.parser.add_argument(
            "--record_path", default="../record.mat", type=str, help="record path"
        )
        self.parser.add_argument(
            "--output_path", default="../results/", type=str, help="output path"
        )
        self.parser.add_argument(
            "--name", type=str, default="", help="name of experiment"
        )

        # GPU
        self.parser.add_argument("--gpu_id", type=str, default="0")
        self.parser.add_argument("--ngpu", type=int, default=1)

        # run
        self.parser.add_argument(
            "--run",
            type=check_arg(str, ["train", "test"]),
            default="train",
            help="train or test",
        )
        self.parser.add_argument("--use_MRF", action="store_true", default=False)

        # train
        self.parser.add_argument("--use_continue", type=str, default="")
        self.parser.add_argument("--epoch_continue", type=int, default=0)
        self.parser.add_argument(
            "--batch_size", type=int, default=5, help="input batch size"
        )
        self.parser.add_argument(
            "--epochs", type=int, default=200, help="number of epochs to train for"
        )
        self.parser.add_argument("--save_freq", type=int, default=50)
        self.parser.add_argument(
            "--lr", type=float, default=0.001, help="learning rate"
        )
        self.parser.add_argument("--lr_decay_ep", type=float, default=100)
        self.parser.add_argument("--lr_decay_gamma", type=float, default=0.1)
        self.parser.add_argument("--lr_decay_method", type=str, default="exp")
        self.parser.add_argument("--optimizer", type=str, default="adam")
        self.parser.add_argument("--k_init", type=str, default="glorot_uniform")
        self.parser.add_argument("--locLambda", type=float, default=0.0)
        self.parser.add_argument("--locT", type=float, default=0.0)
        self.parser.add_argument("--varLambda", type=float, default=0.0)
        self.parser.add_argument("--notinv", action="store_true", default=False)
        self.parser.add_argument("--hmmax", action="store_true", default=False)
        self.parser.add_argument("--correct", action="store_true", default=False)
        self.parser.add_argument("--minsig", type=float, default=0.0)
        self.parser.add_argument("--gan_coef", type=float, default=0.0)
        self.parser.add_argument("--gan_coefdt", type=float, default=0.0)
        self.parser.add_argument("--gan_coefdf", type=float, default=0.0)
        self.parser.add_argument(
            "--train_all",
            action="store_true",
            default=False,
            help="use all data to train the model",
        )

        # input
        self.parser.add_argument(
            "--rot", action="store_true", default=False, help="rotation augmentation"
        )
        self.parser.add_argument(
            "--flip", action="store_true", default=False, help="flipping augmentation"
        )
        self.parser.add_argument(
            "--scale",
            type=float,
            default=0.0,
            help="gamma augmentation, would be ignored if --norm is false",
        )
        self.parser.add_argument(
            "--zoom", type=float, default=2.0, help="probability of zoom augmentation"
        )
        self.parser.add_argument(
            "--zoom_factor",
            type=float,
            default=0.6,
            help="if zoom > 1 the zoom_factor is fixed, otherwise it would be randomly sampled from U[1/zoom_factor, zoom_factor]",
        )
        self.parser.add_argument(
            "--norm",
            action="store_true",
            default=False,
            help="normalization with 99th percentile",
        )
        self.parser.add_argument("--joint", type=str, default="all")  # 15
        self.parser.add_argument("--crop_size", type=str, default="64,64,64")
        self.parser.add_argument("--downsample", action="store_true", default=False)
        self.parser.add_argument(
            "--downsample_label", action="store_true", default=False
        )

        # heat map
        self.parser.add_argument("--sigma", type=float, default=2.0)
        self.parser.add_argument("--mag", type=float, default=10.0)

        # network structure
        self.parser.add_argument("--network", type=str, default="unet")
        self.parser.add_argument("--nStacks", type=int, default=1)
        self.parser.add_argument("--depth", type=int, default=3)
        self.parser.add_argument("--nFeat", type=int, default=64)
        self.parser.add_argument(
            "--normlayer", type=check_arg(str, ["bn", "in", "none"]), default="bn"
        )
        self.parser.add_argument("--network_d", type=str, default="")

        self.initialized = True

    def parse(self, is_predict=False):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        # gpu
        if self.opt.ngpu:
            self.opt.gpu_id = get_gpu(self.opt.ngpu, 11000)
        else:
            self.opt.ngpu = len(self.opt.gpu_id.split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.opt.gpu_id)
        # heatmap zoom
        self.opt.fac = 2 if self.opt.network == "simple" else 1
        # crop size
        self.opt.crop_size = [int(s) for s in self.opt.crop_size.split(",")]
        # time
        self.opt.time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # joint
        if self.opt.joint == "all":
            self.opt.joint = list(range(15))
        else:
            self.opt.joint = [(int(j) - 1) for j in self.opt.joint.split(",")]
        self.opt.nJoint = len(self.opt.joint)
        # name
        if self.opt.name == "":
            self.opt.name = self.opt.time
        # ep continue
        if self.opt.use_continue and self.opt.epoch_continue < 0:
            self.opt.epoch_continue = max(
                [
                    int("".join(filter(str.isdigit, x)))
                    for x in os.listdir(
                        os.path.join(self.opt.output_path, self.opt.use_continue)
                    )
                    if x.endswith(".ckpt.meta")
                ]
            )

        if is_predict:
            return self.opt

        expr_dir = os.path.join(self.opt.output_path, self.opt.name)
        if self.opt.run == "test":
            # load from disk
            self.opt = load_yaml(
                os.path.join(expr_dir, "opt.yaml"),
                self.opt,
                key_to_drop=["run", "use_MRF", "epochs", "epoch_continue"],
            )
        else:
            # save to disk
            mkdir(expr_dir)
            save_yaml(
                os.path.join(expr_dir, "opt.yaml"),
                self.opt,
                key_to_drop=["gpu_id", "ngpu", "save_freq", "buffer_size"],
            )
            copyfiles("./", os.path.join(expr_dir, "backup_code"), "*.py")
            copyfiles("./", os.path.join(expr_dir, "backup_code"), "*.sh")

        print_args(self.opt)

        return self.opt
