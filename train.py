import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from Utils import Rec_Loss, Augmentation, GaussPSF, NYUDataset
from Models import Dense_ASPP_SA
from tqdm import tqdm
import argparse
import os
import random
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Unsupervised DFF')
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--wd', type=float, default=5e-5)
parser.add_argument('--batch-size', type=int, default=3)
parser.add_argument('--nof-iter', type=int, default=4000000)
parser.add_argument('--nof-focus', type=int, default=2)
parser.add_argument('--focal-length', type=float, default=35e-3)
parser.add_argument('--near', type=float, default=1e-3)
parser.add_argument('--far', type=float, default=10)
parser.add_argument('--scale', type=float, default=4)
parser.add_argument('--in-scale', type=int, default=2)
parser.add_argument('--out-scale', type=int, default=1)
args = parser.parse_args()

manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

cudnn.benchmark = True

run_name = 'NYU_unsup_F{}'.format(args.nof_focus)

abs_path = os.path.join(os.path.abspath('Results'), run_name)
if not os.path.exists(abs_path):
    os.makedirs(abs_path)

# Data loading code
train_transform = Augmentation()

NYU_dataset = NYUDataset('path/to/NYU/', transforms=train_transform, scale=args.out_scale, scale2=args.in_scale)
NYU_dataloader = DataLoader(NYU_dataset, torch.cuda.device_count()*args.batch_size, shuffle=True, num_workers=3, drop_last=True)

net = Dense_ASPP_SA()
generator = GaussPSF(7, near=args.near, far=args.far, scale=args.scale)
loss_all = Rec_Loss()

net = torch.nn.DataParallel(net)
generator = torch.nn.DataParallel(generator)
loss_all = torch.nn.DataParallel(loss_all)

net = net.cuda()
generator = generator.cuda()
loss_all = loss_all.cuda()

optimizer  = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)

focal_seq = [0.2, 0.8, 0.1, 0.9, 0.3, 0.7, 0.4, 0.6, 0.5, 0.35]

# Train
net.train()
net.module.freeze_bn()

_iter = iter(NYU_dataloader)

iterator = tqdm(range(0, args.nof_iter),
                total=args.nof_iter,
                leave=False,
                dynamic_ncols=True,
                unit='Batch',
                desc='NYU - ' + run_name)
for index in iterator:
    try:
        gt_depth, image, image_small = next(_iter)
    except:
        _iter = iter(NYU_dataloader)
        gt_depth, image, image_small = next(_iter)

    batch_size = image.size(0)

    image = image.cuda()
    image_small = image_small.cuda()

    # Depth prediction
    pred_depth = net(image_small, image.shape[2:])

    # Focus
    rec_loss = 0
    ssim_loss = 0
    sm_loss = 0
    sharp_loss = 0

    focuses = []
    pred_focuses = []

    aperture = torch.Tensor([args.apt] * batch_size).float().cuda()
    focal_length = torch.Tensor([args.focal_length] * batch_size).float().cuda()

    for fd in range(args.nof_focus):
        focal_depth = torch.Tensor([focal_seq[fd]] * batch_size).float().cuda()
        focused = generator(image, gt_depth, focal_depth, aperture, focal_length)
        pred_focused = generator(image, pred_depth, focal_depth, aperture, focal_length)

        rec_loss_fd, ssim_loss_fd, sm_loss_fd, sharp_loss_fd = loss_all(image, focused, pred_focused, pred_depth)

        rec_loss = rec_loss + rec_loss_fd.mean()
        ssim_loss = ssim_loss + ssim_loss_fd.mean()
        sm_loss = sm_loss + sm_loss_fd.mean()
        sharp_loss = sharp_loss + sharp_loss_fd.mean()

        focuses.append(focused)
        pred_focuses.append(pred_focused)

    rec_loss = rec_loss / args.nof_focus
    ssim_loss = ssim_loss / args.nof_focus
    sm_loss = sm_loss / args.nof_focus
    sharp_loss = sharp_loss / args.nof_focus

    loss = rec_loss + 1e-3*sm_loss + 1e-1*sharp_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    iterator.set_postfix(loss='{:.4f}'.format(loss.item()),
                         rec_loss='{:.4f}'.format(rec_loss.item()),
                         ssim_loss='{:.4f}'.format(ssim_loss.item()))

    if index % 1000 == 0:
        depth_all = np.array(pred_depth.tolist())
        depth_all_gt = np.array(gt_depth.tolist())

        depth_all_gt = (depth_all_gt * 255).astype(np.uint8)
        depth_all = (depth_all * 255).astype(np.uint8)

        for i in range(len(depth_all[:2])):
            cv2.imwrite(abs_path + '/{}_{}_depth_{}_org.png'.format(run_name, index, i), depth_all[i])
            cv2.imwrite(abs_path + '/{}_{}_depth_{}_gt.png'.format(run_name, index, i), depth_all_gt[i])

        torch.save(net.module.state_dict(), '{}.pth'.format(run_name))

