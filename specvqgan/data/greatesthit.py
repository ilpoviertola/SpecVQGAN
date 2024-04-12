import csv
import os
import pickle
from pathlib import Path

import torch
import numpy as np

import sys

sys.path.append(".")
from specvqgan.modules.losses.vggishish.transforms import Crop
from train import instantiate_from_config


class CropImage(Crop):
    def __init__(self, *crop_args):
        super().__init__(*crop_args)


class CropFeats(Crop):
    def __init__(self, *crop_args):
        super().__init__(*crop_args)

    def __call__(self, item):
        item["feature"] = self.preprocessor(image=item["feature"])["image"]
        return item


class GHFeats(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        rgb_feats_dir_path,
        flow_feats_dir_path,
        feat_len,
        feat_depth,
        feat_crop_len,
        replace_feats_with_random,
        random_crop,
        split_path,
        meta_path="/home/hdd/ilpo/datasets/greatesthit/vis-data-256_h264_video_25fps_256side_24000hz_aac_len_5_splitby_random/metadata.csv",
        for_which_class=None,
        feat_sampler_cfg=None,
    ):
        super().__init__()
        self.split = split
        self.meta_path = meta_path
        self.rgb_feats_dir_path = rgb_feats_dir_path
        self.flow_feats_dir_path = flow_feats_dir_path
        self.feat_len = feat_len
        self.feat_depth = feat_depth
        self.feat_crop_len = feat_crop_len
        self.split_path = split_path
        self.feat_sampler_cfg = feat_sampler_cfg
        self.replace_feats_with_random = replace_feats_with_random

        gh_meta = list(csv.reader(open(meta_path), quotechar='"'))[1:]  # skip header
        unique_classes = sorted(list(set(row[5] for row in gh_meta)))
        self.label2target = {
            label: target for target, label in enumerate(unique_classes)
        }
        self.target2label = {
            target: label for label, target in self.label2target.items()
        }
        self.video2target = {row[0]: self.label2target[row[5]] for row in gh_meta}

        if not os.path.exists(split_path):
            raise NotImplementedError(
                f"The splits with clips shoud be available @ {split_path}"
            )

        self.dataset = []
        with open(split_path, encoding="utf-8") as f:
            within_split = f.read().splitlines()

        for basename in within_split:
            files = self._get_all_files_with_same_basename(
                basename, Path(rgb_feats_dir_path)
            )
            self.dataset += files

        if for_which_class:
            raise NotImplementedError

        self.feats_transforms = CropFeats([feat_crop_len, feat_depth], random_crop)

        # ResampleFrames
        self.feat_sampler = (
            None
            if feat_sampler_cfg is None
            else instantiate_from_config(feat_sampler_cfg)
        )

    def _get_all_files_with_same_basename(self, basename: str, data_dir: Path) -> list:
        all_files = (
            data_dir.glob(f"{basename}_denoised*")
            if self.split != "predict"
            else data_dir.glob(f"{basename}*")
        )
        return [
            f.name for f in list(all_files) if f.name.endswith("_resnet50.pkl")
        ]  # return only filenames

    def __getitem__(self, idx):
        item = dict()
        video_clip_name = self.dataset[idx]
        # '/path/zyTX_1BXKDE_16000_26000_mel.npy' -> 'zyTX_1BXKDE_16000_26000'
        # video_clip_name = Path(item['file_path_']).stem.replace('_mel.npy', '')
        video_name = Path(video_clip_name).name.replace("_resnet50.pkl", ".mp4")

        rgb_path = os.path.join(self.rgb_feats_dir_path, f"{video_clip_name}")
        if self.replace_feats_with_random:
            rgb_feats = np.random.rand(self.feat_len, self.feat_depth // 2).astype(
                np.float32
            )
        else:
            rgb_feats = pickle.load(open(rgb_path, "rb"), encoding="bytes")
        feats = rgb_feats
        item["file_path_"] = (rgb_path,)

        # also preprocess flow
        if self.flow_feats_dir_path is not None:
            raise NotImplementedError
            flow_path = os.path.join(self.flow_feats_dir_path, f"{video_clip_name}.pkl")
            # just a dummy random features acting like a fake interface for no features experiment
            if self.replace_feats_with_random:
                flow_feats = np.random.rand(self.feat_len, self.feat_depth // 2).astype(
                    np.float32
                )
            else:
                flow_feats = pickle.load(open(flow_path, "rb"), encoding="bytes")
            # (T, 2*D)
            feats = np.concatenate((rgb_feats, flow_feats), axis=1)
            item["file_path_"] = (rgb_path, flow_path)

        # pad or trim
        feats_padded = np.zeros((self.feat_len, feats.shape[1]))
        feats_padded[: feats.shape[0], :] = feats[: self.feat_len, :]
        item["feature"] = feats_padded

        target = self.video2target[video_name]
        item["target"] = target
        item["label"] = self.target2label[target]

        if self.feats_transforms is not None:
            item = self.feats_transforms(item)

        if self.feat_sampler is not None:
            item = self.feat_sampler(item)

        return item

    def __len__(self):
        return len(self.dataset)


class GHFeatsTrain(GHFeats):
    def __init__(self, condition_dataset_cfg):
        super().__init__("train", **condition_dataset_cfg)


class GHFeatsValidation(GHFeats):
    def __init__(self, condition_dataset_cfg):
        super().__init__("valid", **condition_dataset_cfg)


class GHFeatsTest(GHFeats):
    def __init__(self, condition_dataset_cfg):
        super().__init__("test", **condition_dataset_cfg)


class GHSpecs(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        spec_dir_path,
        mel_num=None,
        spec_len=None,
        spec_crop_len=None,
        random_crop=None,
        crop_coord=None,
        for_which_class=None,
    ):
        super().__init__()
        self.split = split
        self.spec_dir_path = spec_dir_path
        # fixing split_path in here because of compatibility with vggsound which hangles it in vggishish
        self.split_path = f"./data/greatesthit_{split}.txt"
        self.feat_suffix = "_mel.npy"

        if not os.path.exists(self.split_path):
            raise FileNotFoundError

        self.dataset = []
        with open(self.split_path, encoding="utf-8") as f:
            within_split = f.read().splitlines()

        for basename in within_split:
            files = self._get_all_files_with_same_basename(
                basename, Path(spec_dir_path)
            )
            self.dataset += files

        meta_path = "/home/hdd/ilpo/datasets/greatesthit/vis-data-256_h264_video_25fps_256side_24000hz_aac_len_5_splitby_random/metadata.csv"

        gh_meta = list(csv.reader(open(meta_path), quotechar='"'))[1:]  # skip header
        unique_classes = sorted(list(set(row[5] for row in gh_meta)))
        self.label2target = {
            label: target for target, label in enumerate(unique_classes)
        }
        self.target2label = {
            target: label for label, target in self.label2target.items()
        }
        self.video2target = {row[0]: self.label2target[row[5]] for row in gh_meta}

        self.transforms = CropImage([mel_num, spec_crop_len], random_crop)

    def _get_all_files_with_same_basename(self, basename: str, data_dir: Path) -> list:
        all_files = (
            data_dir.glob(f"{basename}_denoised*")
            if self.split != "predict"
            else data_dir.glob(f"{basename}*")
        )
        return [
            f.name for f in list(all_files) if f.name.endswith("_mel.npy")
        ]  # return only filenames

    def __getitem__(self, idx):
        item = {}

        spec_path = self.dataset[idx]

        spec = np.load(f"{self.spec_dir_path}/{spec_path}")
        item["input"] = spec
        item["file_path_"] = spec_path

        item["target"] = self.video2target[
            str(Path(spec_path).name.replace("_mel.npy", ".mp4"))
        ]
        item["label"] = self.target2label[item["target"]]

        if self.transforms is not None:
            item = self.transforms(item)

        # specvqgan expects `image` and `file_path_` keys in the item
        # it also expects inputs in [-1, 1] but specs are in [0, 1]
        item["image"] = 2 * item["input"] - 1
        item.pop("input")

        return item

    def __len__(self):
        return len(self.dataset)


class GHSpecsCondOnFeats(torch.utils.data.Dataset):

    def __init__(self, split, specs_dataset_cfg, condition_dataset_cfg):
        self.specs_dataset_cfg = specs_dataset_cfg
        self.condition_dataset_cfg = condition_dataset_cfg

        self.specs_dataset = GHSpecs(split, **specs_dataset_cfg)
        self.feats_dataset = GHFeats(split, **condition_dataset_cfg)
        assert len(self.specs_dataset) == len(self.feats_dataset)

    def __getitem__(self, idx):
        specs_item = self.specs_dataset[idx]
        feats_item = self.feats_dataset[idx]

        # sanity check and removing those from one of the dicts
        for key in ["target", "label"]:
            assert (
                specs_item[key] == feats_item[key]
            ), f"specs: {specs_item[key]}, feats: {feats_item[key]}"
            feats_item.pop(key)

        # keeping both sets of paths to features
        specs_item["file_path_specs_"] = specs_item.pop("file_path_")
        feats_item["file_path_feats_"] = feats_item.pop("file_path_")

        # merging both dicts
        specs_feats_item = dict(**specs_item, **feats_item)

        return specs_feats_item

    def __len__(self):
        return len(self.specs_dataset)


class GHSpecsCondOnFeatsTrain(GHSpecsCondOnFeats):
    def __init__(self, specs_dataset_cfg, condition_dataset_cfg):
        super().__init__("train", specs_dataset_cfg, condition_dataset_cfg)


class GHSpecsCondOnFeatsValidation(GHSpecsCondOnFeats):
    def __init__(self, specs_dataset_cfg, condition_dataset_cfg):
        super().__init__("valid", specs_dataset_cfg, condition_dataset_cfg)


class GHSpecsCondOnFeatsTest(GHSpecsCondOnFeats):
    def __init__(self, specs_dataset_cfg, condition_dataset_cfg):
        super().__init__("test", specs_dataset_cfg, condition_dataset_cfg)


if __name__ == "__main__":
    import sys

    sys.path.append(".")
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("./configs/greatesthit_transformer.yaml")
    data = instantiate_from_config(cfg.data)
    data.prepare_data()
    data.setup()
    print(len(data.datasets["train"]))
    print(data.datasets["train"][24])
    print(data.datasets["validation"][24])
    print(data.datasets["test"][24])
    print(data.datasets["train"][24]["feature"].shape)
