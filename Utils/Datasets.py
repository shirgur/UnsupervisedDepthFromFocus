import torch
from torch.utils.data import Dataset
import numpy as np
from scipy import io
from path import Path
import h5py
import cv2
import glob


class NYUDataset(Dataset):
    def __init__(self, path, transforms=None, scale=1, scale2=2, extra=50000):
        self.transforms = transforms
        self.scale = scale
        self.scale2 = scale2

        classes = {p.split('/')[-1].split('_0')[0] for p in glob.glob(path + 'raw_data/*_out')}
        sub_select = extra // len(classes)
        res_select = extra % len(classes)

        tmp = []

        for cls in classes:
            tmp.append([cls, len(glob.glob(path + 'raw_data/{}**_out/*rgb.png'.format(cls)))])
        classes = sorted(tmp, key=lambda x: x[1])

        self.images = []
        self.depths = []

        _ext = 0
        for i, cls in enumerate(classes):
            rem_cls = len(classes) - i
            images = glob.glob(path + 'raw_data/{}**_out/*rgb.png'.format(cls[0]))
            depths = glob.glob(path + 'raw_data/{}**_out/*depth.png'.format(cls[0]))
            images = sorted(images)
            depths = sorted(depths)
            if i < len(classes) - 1:
                if sub_select + _ext // rem_cls <= len(images):
                    toselect = sub_select + _ext // rem_cls
                    indices = np.random.choice(len(images), toselect, replace=False)
                    self.images += np.array(images)[indices].tolist()
                    self.depths += np.array(depths)[indices].tolist()
                    _ext = max(0, _ext - _ext // rem_cls)
                else:
                    self.images += images
                    self.depths += depths
                    _ext += sub_select - len(images)
            else:
                if sub_select + _ext + res_select <= len(images):
                    toselect = sub_select + _ext + res_select
                    indices = np.random.choice(len(images), toselect, replace=False)
                    self.images += np.array(images)[indices].tolist()
                    self.depths += np.array(depths)[indices].tolist()
                else:
                    self.images += images
                    self.depths += depths

        self.images = sorted(self.images)
        self.depths = sorted(self.depths)

        assert len(self.images) == len(self.depths)

        splits = io.loadmat(path + 'splits.mat')
        train_idx = np.array(splits['trainNdxs']).squeeze(-1) - 1

        _h5py = h5py.File(path + 'nyu_depth_v2_labeled.mat', 'r')
        self.images_file = np.transpose(_h5py['images'], (0, 3, 2, 1))
        self.depths_file = np.transpose(_h5py['depths'], (0, 2, 1))
        _h5py.close()

        self.images_file = self.images_file[train_idx]
        self.depths_file = self.depths_file[train_idx]

    def __len__(self):
        return len(self.images) + len(self.images_file)

    def __getitem__(self, idx):

        if idx < len(self.images_file):
            img_gt_depth = self.depths_file[idx]
            img_org = self.images_file[idx]

        else:
            idx -= len(self.images_file)
            img_gt_depth = np.clip(cv2.imread(self.depths[idx])[:, :, 0] / 10, 0, 10)
            img_org = cv2.imread(self.images[idx])

        img_gt_depth = cv2.resize(img_gt_depth, (640//self.scale, 480//self.scale))
        img_org = cv2.resize(img_org, (640//self.scale, 480//self.scale))

        if self.transforms is not None:
            img_org, img_gt_depth = self.transforms(img_org, img_gt_depth)

        img_org_small = cv2.resize(img_org, (640//self.scale2, 480//self.scale2))

        img_org = torch.from_numpy(img_org).permute(2, 0, 1).float() / 255
        img_gt_depth = torch.from_numpy(img_gt_depth).float() / 10
        img_org_small = torch.from_numpy(img_org_small).permute(2, 0, 1).float() / 255

        img_gt_depth.requires_grad = False
        img_org.requires_grad = False
        img_org_small.requires_grad = False

        return img_gt_depth, img_org, img_org_small


class KITTIDataset(Dataset):
    def __init__(self, root, list_file='train.txt', transform=None, scale=1, scale2=2):
        super(KITTIDataset, self).__init__()
        self.root = Path(root)
        scene_list_path = self.root/list_file
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.scale = scale
        self.scale2= scale2

        self.gt_depth = []
        self.org = []

        for scene in self.scenes:
            imgs = sorted(scene.files('*.jpg'))
            depths = sorted(scene.files('*.npy'))
            self.org += imgs
            self.gt_depth += depths

        assert len(self.gt_depth) == len(self.org)

    def __getitem__(self, idx):

        img_gt_depth = np.load(self.gt_depth[idx])
        img_org = cv2.imread(self.org[idx])
        H, W, _ = img_org.shape

        if self.scale > 1:
            img_gt_depth = cv2.resize(img_gt_depth, (W // self.scale, H // self.scale))
            img_org = cv2.resize(img_org, (W // self.scale, H // self.scale))

        if self.transform is not None:
            img_org, img_gt_depth = self.transform(img_org, img_gt_depth)

        img_org_small = cv2.resize(img_org, (W // self.scale2, H // self.scale2))

        img_org = torch.from_numpy(img_org).permute(2, 0, 1).float() / 255
        img_gt_depth = torch.from_numpy(img_gt_depth).float() / 80
        img_org_small = torch.from_numpy(img_org_small).permute(2, 0, 1).float() / 255

        img_gt_depth.requires_grad = False
        img_org.requires_grad = False
        img_org_small.requires_grad = False

        return img_gt_depth, img_org, img_org_small

    def __len__(self):
        return len(self.org)


class Make3DDataset(Dataset):
    def __init__(self, path, transforms=None, scale1=1, scale2=2):
        self.transforms = transforms
        self.scale1 = scale1
        self.scale2 = scale2

        train_images = glob.glob(path + '/Train/*.jpg')
        train_depth = glob.glob(path + '/Train/*.mat')

        self.train_images = sorted(train_images, key=lambda p: p.split('/')[-1].split('img-')[-1])
        self.train_depth = sorted(train_depth, key=lambda p: p.split('/')[-1].split('depth_sph_corr-')[-1])

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        image = cv2.imread(self.train_images[idx])
        depth = io.loadmat(self.train_depth[idx])['Position3DGrid'][:, :, 3]

        image = cv2.resize(image, (460 // self.scale1, 345 // self.scale1), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (460 // self.scale1, 345 // self.scale1), interpolation=cv2.INTER_LINEAR)

        if self.transforms is not None:
            image, depth = self.transforms(image, depth)

        image_small = cv2.resize(image, (460 // self.scale2, 345 // self.scale2))

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255
        image_small = torch.from_numpy(image_small).permute(2, 0, 1).float() / 255
        depth = torch.from_numpy(depth).float() / 80

        depth.requires_grad = False
        image.requires_grad = False
        image_small.requires_grad = False

        return depth, image, image_small