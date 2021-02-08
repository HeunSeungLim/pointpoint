import os
import sys
import argparse
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import utils
from utils import PointLoss
from utils import distance_squre
import data_utils as d_utils
import ModelNet40Loader
import shapenet_part_loader
from model_PFNet import _netlocalD, _netG
# YH
import itertools
from pytictoc import TicToc


from statistics import mean
#
tic_toc = TicToc()

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='dataset/train', help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=48, help='input batch size')
parser.add_argument('--pnum', type=int, default=2048, help='the point number of a sample')
parser.add_argument('--crop_point_num', type=int, default=512, help='0 means do not use else use with this weight')
# YH
parser.add_argument('--crop_point_num_b2a', type=int, default=1536, help='0 means do not use else use with this weight')
#
parser.add_argument('--nc', type=int, default=3)

parser.add_argument('--niter', type=int, default=201, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', type=bool, default=False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--D_choose', type=int, default=1, help='0 not use D-net,1 use D-net')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
# YH
parser.add_argument('--netG_b2a', default='', help="path to netG_b2a (to continue training")
parser.add_argument('--netD_b', default='', help="path to netD_b (to continue training")
#
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--drop', type=float, default=0.2)
parser.add_argument('--num_scales', type=int, default=3, help='number of scales')
parser.add_argument('--point_scales_list', type=list, default=[2048, 1024, 512], help='number of points in each scales')
parser.add_argument('--each_scales_size', type=int, default=1, help='each scales size')
parser.add_argument('--wtl2', type=float, default=0.95, help='0 means do not use else use with this weight')
parser.add_argument('--cropmethod', default='random_center', help='random|center|random_center')
opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USE_CUDA = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# point_netG = _netG(opt.num_scales,opt.each_scales_size,opt.point_scales_list,opt.crop_point_num)
# point_netD = _netlocalD(opt.crop_point_num)
# YH
netG_a2b = _netG(opt.num_scales, opt.each_scales_size, opt.point_scales_list, opt.crop_point_num)
netG_b2a = _netG(opt.num_scales, opt.each_scales_size, opt.point_scales_list, opt.crop_point_num_b2a)
netD_a = _netlocalD(opt.crop_point_num)
netD_b = _netlocalD(opt.crop_point_num_b2a)
#
cudnn.benchmark = True
resume_epoch = 0


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Conv1d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


if USE_CUDA:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # YH
    netG_a2b = torch.nn.DataParallel(netG_a2b)
    netD_a = torch.nn.DataParallel(netD_a)
    netG_b2a = torch.nn.DataParallel(netG_b2a)
    netD_b = torch.nn.DataParallel(netD_b)
    netG_a2b.to(device)
    netG_a2b.apply(weights_init_normal)
    netD_a.to(device)
    netD_a.apply(weights_init_normal)
    netG_b2a.to(device)
    netG_b2a.apply(weights_init_normal)
    netD_b.to(device)
    netD_b.apply(weights_init_normal)
    # YH
    # point_netG = torch.nn.DataParallel(point_netG)
    # point_netD = torch.nn.DataParallel(point_netD)
    # point_netG.to(device)
    # point_netG.apply(weights_init_normal)
    # point_netD.to(device)
    # point_netD.apply(weights_init_normal)
if opt.netG != '':
    # YH
    netG_a2b.load_state_dict(torch.load(opt.netG, map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netG)['epoch']
    #
    # point_netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
    # resume_epoch = torch.load(opt.netG)['epoch']
if opt.netD != '':
    # YH
    netD_a.load_state_dict(torch.load(opt.netD, map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netD)['epoch']
    #
    # point_netD.load_state_dict(torch.load(opt.netD,map_location=lambda storage, location: storage)['state_dict'])
    # resume_epoch = torch.load(opt.netD)['epoch']
# YH
if opt.netG_b2a != '':
    netG_b2a.load_state_dict(torch.load(opt.netG_b2a, map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netG_b2a)['epoch']
if opt.netD_b != '':
    netD_b.load_state_dict(torch.load(opt.netD_b, map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netD_b)['epoch']
#

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

transforms = transforms.Compose(
    [
        d_utils.PointcloudToTensor(),
    ]
)
dset = shapenet_part_loader.PartDataset(root='../HL_Cycle/dataset/shapenetcore_partanno_segmentation_benchmark_v0/',
                                        classification=True, class_choice=None, npoints=opt.pnum, split='train')
assert dset
dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

test_dset = shapenet_part_loader.PartDataset(
    root='../HL_Cycle/dataset/shapenetcore_partanno_segmentation_benchmark_v0/', classification=True, class_choice=None,
    npoints=opt.pnum, split='test')
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize,
                                              shuffle=True, num_workers=int(opt.workers))

# dset = ModelNet40Loader.ModelNet40Cls(opt.pnum, train=True, transforms=transforms, download = False)
# assert dset
# dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batchSize,
#                                         shuffle=True,num_workers = int(opt.workers))
#
#
# test_dset = ModelNet40Loader.ModelNet40Cls(opt.pnum, train=False, transforms=transforms, download = False)
# test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize,
#                                         shuffle=True,num_workers = int(opt.workers))

# pointcls_net.apply(weights_init)
# YH
print("######netG_a2b######")
print(netG_a2b)
print("######netD_a######")
print(netD_a)
print("######netG_b2a######")
print(netG_b2a)
print("######netD_b######")
print(netD_b)
#
# print(point_netG)
# print(point_netD)

criterion = torch.nn.BCEWithLogitsLoss().to(device)
criterion_PointLoss = PointLoss().to(device)

# setup optimizer
# YH
paramsG_a2b = netG_a2b.parameters()
paramsG_b2a = netG_b2a.parameters()
paramsD_a = netD_a.parameters()
paramsD_b = netD_b.parameters()
optimG = torch.optim.Adam(itertools.chain(paramsG_a2b, paramsG_b2a), lr=0.0001, betas=(0.9, 0.999), eps=1e-05,
                          weight_decay=opt.weight_decay)
optimD = torch.optim.Adam(itertools.chain(paramsD_a, paramsD_b), lr=0.0001, betas=(0.9, 0.999), eps=1e-05,
                          weight_decay=opt.weight_decay)
schedulerG = torch.optim.lr_scheduler.StepLR(optimG, step_size=40, gamma=0.2)
schedulerD = torch.optim.lr_scheduler.StepLR(optimD, step_size=40, gamma=0.2)
#
# optimizerD = torch.optim.Adam(point_netD.parameters(), lr=0.0001,betas=(0.9, 0.999),eps=1e-05,weight_decay=opt.weight_decay)
# optimizerG = torch.optim.Adam(point_netG.parameters(), lr=0.0001,betas=(0.9, 0.999),eps=1e-05 ,weight_decay=opt.weight_decay)
# schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=40, gamma=0.2)
# schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=40, gamma=0.2)

real_label = 1
fake_label = 0

crop_point_num = int(opt.crop_point_num)
# YH
crop_point_num_b2a = int(opt.crop_point_num_b2a)
#
input_cropped1 = torch.FloatTensor(opt.batchSize, opt.pnum, 3)
label = torch.FloatTensor(opt.batchSize)

num_batch = len(dset) / opt.batchSize
if __name__ == '__main__':
    ###########################
    #  G-NET and T-NET
    ##########################
    if opt.D_choose == 1:
        for epoch in range(resume_epoch, opt.niter):
            if epoch < 30:
                alpha1 = 0.01
                alpha2 = 0.02
            elif epoch < 80:
                alpha1 = 0.05
                alpha2 = 0.1
            else:
                alpha1 = 0.1
                alpha2 = 0.2

            for i, data in enumerate(dataloader, 0):

                tic_toc.tic()  # 시작시간
                real_point, target = data
                # YH
                real_point_b = real_point
                #

                batch_size = real_point.size()[0]
                real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)
                real_center_b = torch.FloatTensor(batch_size, 1, opt.crop_point_num_b2a, 3)  # YH
                input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)
                input_cropped1 = input_cropped1.data.copy_(real_point)
                input_cropped1_b = torch.FloatTensor(batch_size, opt.pnum, 3)  # YH
                input_cropped1_b = input_cropped1_b.data.copy_(real_point)  # YH
                real_point = torch.unsqueeze(real_point, 1)
                input_cropped1 = torch.unsqueeze(input_cropped1, 1)
                input_cropped1_b = torch.unsqueeze(input_cropped1_b, 1)  # YH
                p_origin = [0, 0, 0]
                if opt.cropmethod == 'random_center':
                    # Set viewpoints
                    choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]),
                              torch.Tensor([-1, 0, 0]), torch.Tensor([-1, 1, 0])]
                    for m in range(batch_size):
                        index = random.sample(choice, 1)  # Random choose one of the viewpoint
                        distance_list = []
                        p_center = index[0]
                        for n in range(opt.pnum):
                            distance_list.append(distance_squre(real_point[m, 0, n], p_center))
                            input_cropped1_b.data[m, 0, n, :] = torch.FloatTensor([0, 0, 0])  # YH
                        distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])

                        for sp in range(opt.crop_point_num):
                            input_cropped1.data[m, 0, distance_order[sp][0]] = torch.FloatTensor([0, 0, 0])
                            # input_cropped1_b.data[m,0,distance_order[sp][0]] = real_point[m,0,distance_order[sp][0]] # YH
                            real_center.data[m, 0, sp] = real_point[m, 0, distance_order[sp][0]]
                        for sp in range(opt.crop_point_num_b2a):  # YH
                            input_cropped1_b.data[m, 0, distance_order[sp][0]] = real_point[
                                m, 0, distance_order[512 + sp][0]]  # YH
                            real_center_b.data[m, 0, sp] = real_point[m, 0, distance_order[512 + sp][0]]  # YH
                label.resize_([batch_size, 1]).fill_(real_label)
                real_point = real_point.to(device)
                real_center = real_center.to(device)
                real_center_b = real_center_b.to(device)  # YH
                input_cropped1 = input_cropped1.to(device)
                input_cropped1_b = input_cropped1_b.to(device)  # YH
                label = label.to(device)
                ############################
                # (1) data prepare
                ###########################
                real_center = Variable(real_center, requires_grad=True)
                real_center = torch.squeeze(real_center, 1)
                real_center_key1_idx = utils.farthest_point_sample(real_center, 64, RAN=False)
                real_center_key1 = utils.index_points(real_center, real_center_key1_idx)
                real_center_key1 = Variable(real_center_key1, requires_grad=True)

                real_center_key2_idx = utils.farthest_point_sample(real_center, 128, RAN=True)
                real_center_key2 = utils.index_points(real_center, real_center_key2_idx)
                real_center_key2 = Variable(real_center_key2, requires_grad=True)

                input_cropped1 = torch.squeeze(input_cropped1, 1)
                input_cropped2_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[1], RAN=True)
                input_cropped2 = utils.index_points(input_cropped1, input_cropped2_idx)
                input_cropped3_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[2], RAN=False)
                input_cropped3 = utils.index_points(input_cropped1, input_cropped3_idx)
                input_cropped1 = Variable(input_cropped1, requires_grad=True)
                input_cropped2 = Variable(input_cropped2, requires_grad=True)
                input_cropped3 = Variable(input_cropped3, requires_grad=True)
                input_cropped2 = input_cropped2.to(device)
                input_cropped3 = input_cropped3.to(device)
                input_cropped = [input_cropped1, input_cropped2, input_cropped3]

                # YH
                real_center_b = Variable(real_center_b, requires_grad=True)
                real_center_b = torch.squeeze(real_center_b, 1)

                real_center_b_key1_idx = utils.farthest_point_sample(real_center_b, 64, RAN=False)
                real_center_b_key1 = utils.index_points(real_center_b, real_center_b_key1_idx)
                real_center_b_key1 = Variable(real_center_b_key1, requires_grad=True)

                real_center_b_key2_idx = utils.farthest_point_sample(real_center_b, 128, RAN=True)
                real_center_b_key2 = utils.index_points(real_center_b, real_center_b_key2_idx)
                real_center_b_key2 = Variable(real_center_b_key2, requires_grad=True)

                input_cropped1_b = torch.squeeze(input_cropped1_b, 1)

                input_cropped2_b_idx = utils.farthest_point_sample(input_cropped1_b, opt.point_scales_list[1], RAN=True)
                input_cropped2_b = utils.index_points(input_cropped1_b, input_cropped2_b_idx)

                input_cropped3_b_idx = utils.farthest_point_sample(input_cropped1_b, opt.point_scales_list[2],
                                                                   RAN=False)
                input_cropped3_b = utils.index_points(input_cropped1_b, input_cropped3_b_idx)

                input_cropped1_b = Variable(input_cropped1_b, requires_grad=True)
                input_cropped2_b = Variable(input_cropped2_b, requires_grad=True)
                input_cropped3_b = Variable(input_cropped3_b, requires_grad=True)

                input_cropped2_b = input_cropped2_b.to(device)
                input_cropped3_b = input_cropped3_b.to(device)
                input_cropped_b = [input_cropped1_b, input_cropped2_b, input_cropped3_b]

                netG_a2b = netG_a2b.train()
                netG_b2a = netG_b2a.train()
                netD_a = netD_a.train()
                netD_b = netD_b.train()

                loss_G_a2b_train = []
                loss_G_b2a_train = []
                loss_D_a_train = []
                loss_D_b_train = []
                loss_C_a_train = []
                loss_C_b_train = []
                loss_I_a_train = []
                loss_I_b_train = []


                fake_center1_b, fake_center2_b, fake_b = netG_a2b(input_cropped)    #################################### output_a
                fake_center1_a, fake_center2_a, fake_a = netG_b2a(input_cropped_b)  #################################### output_b

                second_input_a2b_b2a = fake_a
                second_input_b2a_a2b = fake_b

                input_cropped_a2b_b2a = torch.zeros(batch_size, opt.pnum, 3)
                input_cropped_b2a_a2b = torch.zeros(batch_size, opt.pnum, 3)
                total_index_a2b_b2a = second_input_a2b_b2a.shape[1]
                total_index_b2a_a2b = second_input_b2a_a2b.shape[1]
                for m in range(batch_size):
                    for n in range(total_index_a2b_b2a):
                        input_cropped_a2b_b2a[m, n, :] = second_input_a2b_b2a[m, n, :]
                    for o in range(total_index_b2a_a2b):
                        input_cropped_b2a_a2b[m, o, :] = second_input_b2a_a2b[m, o, :]
                label.data.fill_(fake_label)
                input_cropped_a2b_b2a.to(device)
                input_cropped_b2a_a2b.to(device)
                label = label.to(device)

                input_cropped_a2b_b2a_2_idx = utils.farthest_point_sample(input_cropped_a2b_b2a,
                                                                          opt.point_scales_list[1], RAN=True)
                input_cropped_a2b_b2a_2 = utils.index_points(input_cropped_a2b_b2a, input_cropped_a2b_b2a_2_idx)
                input_cropped_a2b_b2a_3_idx = utils.farthest_point_sample(input_cropped_a2b_b2a,
                                                                          opt.point_scales_list[2], RAN=False)
                input_cropped_a2b_b2a_3 = utils.index_points(input_cropped_a2b_b2a, input_cropped_a2b_b2a_3_idx)
                input_cropped_a2b_b2a = Variable(input_cropped_a2b_b2a, requires_grad=True)
                input_cropped_a2b_b2a_2 = Variable(input_cropped_a2b_b2a_2, requires_grad=True)
                input_cropped_a2b_b2a_3 = Variable(input_cropped_a2b_b2a_3, requires_grad=True)
                input_cropped_a2b_b2a_2 = input_cropped_a2b_b2a_2.to(device)
                input_cropped_a2b_b2a_3 = input_cropped_a2b_b2a_3.to(device)
                input_cropped_second_a2b_b2a = [input_cropped_a2b_b2a, input_cropped_a2b_b2a_2, input_cropped_a2b_b2a_3]

                input_cropped_b2a_a2b_2_idx = utils.farthest_point_sample(input_cropped_b2a_a2b,
                                                                          opt.point_scales_list[1], RAN=True)
                input_cropped_b2a_a2b_2 = utils.index_points(input_cropped_b2a_a2b, input_cropped_b2a_a2b_2_idx)
                input_cropped_b2a_a2b_3_idx = utils.farthest_point_sample(input_cropped_b2a_a2b,
                                                                          opt.point_scales_list[2], RAN=False)
                input_cropped_b2a_a2b_3 = utils.index_points(input_cropped_b2a_a2b, input_cropped_b2a_a2b_3_idx)
                input_cropped_b2a_a2b = Variable(input_cropped_b2a_a2b, requires_grad=True)
                input_cropped_b2a_a2b_2 = Variable(input_cropped_b2a_a2b_2, requires_grad=True)
                input_cropped_b2a_a2b_3 = Variable(input_cropped_b2a_a2b_3, requires_grad=True)
                input_cropped_b2a_a2b_2 = input_cropped_b2a_a2b_2.to(device)
                input_cropped_b2a_a2b_3 = input_cropped_b2a_a2b_3.to(device)
                input_cropped_second_b2a_a2b = [input_cropped_b2a_a2b, input_cropped_b2a_a2b_2, input_cropped_b2a_a2b_3]

                ############################
                # (5) YH: Update D Network for second
                ###########################
                # netG_b2a
                fake_center_a2b_b2a_1, fake_center_a2b_b2a_2, fake_center_a2b_b2a = netG_a2b(input_cropped_second_b2a_a2b) #################################### recon_a
                fake_center_b2a_a2b_1, fake_center_b2a_a2b_2, fake_center_b2a_a2b = netG_b2a(input_cropped_second_a2b_b2a) #################################### recon_b

                # fake_center1_b, fake_center2_b, fake_b = netG_a2b(input_cropped)  #################################### output_a
                # fake_center1_a, fake_center2_a, fake_a = netG_b2a(input_cropped_b)  #################################### output_b
                
                utils.set_requires_grad([netD_a, netD_b], False)
                optimG.zero_grad()

                fake_b = torch.unsqueeze(fake_b, 1)
                fake_a = torch.unsqueeze(fake_a, 1)
                real_center_b = torch.unsqueeze(real_center_b, 1)
                real_center = torch.unsqueeze(real_center, 1)

                pred_real_a = netD_a(real_center)
                pred_fake_a = netD_a(fake_b.detach())

                loss_D_a_real = criterion(pred_real_a, torch.ones_like(pred_real_a))
                loss_D_a_fake = criterion(pred_fake_a, torch.zeros_like(pred_fake_a))
                loss_D_a = 0.5 * (loss_D_a_real + loss_D_a_fake)

                pred_real_b = netD_b(real_center_b)
                pred_fake_b = netD_b(fake_a.detach())

                loss_D_b_real = criterion(pred_real_b, torch.ones_like(pred_real_b))
                loss_D_b_fake = criterion(pred_fake_b, torch.zeros_like(pred_fake_b))
                loss_D_b = 0.5 * (loss_D_b_real + loss_D_b_fake)


                # backward netD
                loss_D = loss_D_a + loss_D_b
                loss_D.backward()
                optimD.step()

                utils.set_requires_grad([netD_a, netD_b], False)
                optimG.zero_grad()

                pred_fake_a = netD_a(fake_b)
                pred_fake_b = netD_b(fake_a)

                loss_G_a2b = criterion(pred_fake_b, torch.ones_like(pred_fake_b))
                loss_G_b2a = criterion(pred_fake_a, torch.ones_like(pred_fake_a))

                CD_LOSS_a = criterion_PointLoss(torch.squeeze(fake_a, 1), torch.squeeze(real_center, 1))
                errG_l2_a = criterion_PointLoss(torch.squeeze(fake_a, 1), torch.squeeze(real_center, 1)) \
                            + alpha1 * criterion_PointLoss(fake_center1_a, real_center_key1) \
                            + alpha2 * criterion_PointLoss(fake_center2_a, real_center_key2)
                CD_LOSS_b = criterion_PointLoss(torch.squeeze(fake_b, 1), torch.squeeze(real_center_b, 1))
                errG_l2_b = criterion_PointLoss(torch.squeeze(fake_b, 1), torch.squeeze(real_center_b, 1)) \
                            + alpha1 * criterion_PointLoss(fake_center1_b, real_center_b_key1) \
                            + alpha2 * criterion_PointLoss(fake_center2_b, real_center_b_key2)

                errG_l2_a2b_b2a = criterion_PointLoss(torch.squeeze(fake_center_a2b_b2a, 1),
                                                      torch.squeeze(real_center_b, 1)) \
                                  + alpha1 * criterion_PointLoss(fake_center_a2b_b2a_1, real_center_b_key1) \
                                  + alpha2 * criterion_PointLoss(fake_center_a2b_b2a_2, real_center_b_key2)

                errG_l2_b2a_a2b = criterion_PointLoss(torch.squeeze(fake_center_b2a_a2b, 1),
                                                      torch.squeeze(real_center, 1)) \
                                  + alpha1 * criterion_PointLoss(fake_center_b2a_a2b_1, real_center_key1) \
                                  + alpha2 * criterion_PointLoss(fake_center_b2a_a2b_2, real_center_key2)

                loss_G = (loss_G_a2b + loss_G_b2a) + \
                         (10 * errG_l2_a2b_b2a + 10 * errG_l2_b2a_a2b) + \
                         (10 * errG_l2_a + 10 * errG_l2_b) * 5e-1

                loss_G.backward()
                optimG.step()

                # print
                print('/*########################################*/')
                loss_G_a2b_train += [loss_G_a2b.item()]
                loss_G_b2a_train += [loss_G_b2a.item()]

                loss_D_a_train += [loss_D_a.item()]
                loss_D_b_train += [loss_D_b.item()]

                loss_C_a_train += [errG_l2_a2b_b2a.item()]
                loss_C_b_train += [errG_l2_b2a_a2b.item()]

                # if wgt_i > 0:
                loss_I_a_train += [errG_l2_a.item()]
                loss_I_b_train += [errG_l2_b.item()]

                print('[%d/%d][%d/%d]: '
                      'G_a2b: %.4f G_b2a: %.4f D_a: %.4f D_b: %.4f C_a: %.4f C_b: %.4f I_a: %.4f I_b: %.4f'
                      % (epoch, opt.niter, i, len(dataloader),
                         mean(loss_G_a2b_train), mean(loss_G_b2a_train),
                         mean(loss_D_a_train), mean(loss_D_b_train),
                         mean(loss_C_a_train), mean(loss_C_b_train),
                         mean(loss_I_a_train), mean(loss_I_b_train)))
                # f = open('loss_PFNet.txt', 'a')
                # f.write('\n' + '[%d/%d][%d/%d] Loss_D_a2b_b2a: %.4f Loss_G_a2b_b2a: %.4f / %.4f / %.4f /%.4f'
                #         % (epoch, opt.niter, i, len(dataloader),
                #            errD_fake_a2b_b2a.data, errG_D_a2b_b2a.data, errG_l2_a2b_b2a, errG_a2b_b2a, CD_LOSS_a2b_b2a))
                # f.write('\n' + '[%d/%d][%d/%d] Loss_D_b2a_a2b: %.4f Loss_G_b2a_a2b: %.4f / %.4f / %.4f /%.4f'
                #         % (epoch, opt.niter, i, len(dataloader),
                #            errD_fake_b2a_a2b.data, errG_D_b2a_a2b.data, errG_l2_b2a_a2b, errG_b2a_a2b, CD_LOSS_b2a_a2b))
                # YH Code END

                if i % 10 == 0:
                    print('After, ', i, '-th batch')
                    # f.write('\n' + 'After, ' + str(i) + '-th batch')
                    for i, data in enumerate(test_dataloader, 0):
                        real_point, target = data

                        batch_size = real_point.size()[0]
                        real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)
                        real_center_b = torch.FloatTensor(batch_size, 1, opt.crop_point_num_b2a, 3)  # YH
                        input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)
                        input_cropped1 = input_cropped1.data.copy_(real_point)
                        input_cropped1_b = torch.FloatTensor(batch_size, opt.pnum, 3)  # YH
                        input_cropped1_b = input_cropped1_b.data.copy_(real_point)  # YH
                        real_point = torch.unsqueeze(real_point, 1)
                        input_cropped1 = torch.unsqueeze(input_cropped1, 1)
                        input_cropped1_b = torch.unsqueeze(input_cropped1_b, 1)  # YH

                        p_origin = [0, 0, 0]

                        if opt.cropmethod == 'random_center':
                            choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]),
                                      torch.Tensor([-1, 0, 0]), torch.Tensor([-1, 1, 0])]

                            for m in range(batch_size):
                                index = random.sample(choice, 1)
                                distance_list = []
                                p_center = index[0]
                                for n in range(opt.pnum):
                                    distance_list.append(distance_squre(real_point[m, 0, n], p_center))
                                    input_cropped1_b.data[m, 0, n, :] = torch.FloatTensor([0, 0, 0])  # YH
                                distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])
                                for sp in range(opt.crop_point_num):
                                    input_cropped1.data[m, 0, distance_order[sp][0]] = torch.FloatTensor([0, 0, 0])
                                    real_center.data[m, 0, sp] = real_point[m, 0, distance_order[sp][0]]
                                for sp in range(opt.crop_point_num_b2a):  # YH
                                    input_cropped1_b.data[m, 0, distance_order[sp][0]] = real_point[
                                        m, 0, distance_order[512 + sp][0]]  # YH
                                    real_center_b.data[m, 0, sp] = real_point[m, 0, distance_order[512 + sp][0]]  # YH
                        real_center = real_center.to(device)
                        real_center = torch.squeeze(real_center, 1)
                        real_center_b = real_center_b.to(device)  # YH
                        input_cropped1 = input_cropped1.to(device)
                        input_cropped1 = torch.squeeze(input_cropped1, 1)
                        input_cropped1_b = input_cropped1_b.to(device)  # YH
                        input_cropped1_b = torch.squeeze(input_cropped1_b, 1)  # YH
                        input_cropped2_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[1],
                                                                         RAN=True)
                        input_cropped2 = utils.index_points(input_cropped1, input_cropped2_idx)
                        input_cropped3_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[2],
                                                                         RAN=False)
                        input_cropped3 = utils.index_points(input_cropped1, input_cropped3_idx)
                        input_cropped1 = Variable(input_cropped1, requires_grad=False)
                        input_cropped2 = Variable(input_cropped2, requires_grad=False)
                        input_cropped3 = Variable(input_cropped3, requires_grad=False)
                        input_cropped2 = input_cropped2.to(device)
                        input_cropped3 = input_cropped3.to(device)
                        input_cropped = [input_cropped1, input_cropped2, input_cropped3]
                        # YH
                        input_cropped2_b_idx = utils.farthest_point_sample(input_cropped1_b, opt.point_scales_list[1],
                                                                           RAN=True)
                        input_cropped2_b = utils.index_points(input_cropped1_b, input_cropped2_b_idx)
                        input_cropped3_b_idx = utils.farthest_point_sample(input_cropped1_b, opt.point_scales_list[2],
                                                                           RAN=False)
                        input_cropped3_b = utils.index_points(input_cropped1_b, input_cropped3_b_idx)
                        input_cropped1_b = Variable(input_cropped1_b, requires_grad=False)
                        input_cropped2_b = Variable(input_cropped2_b, requires_grad=False)
                        input_cropped3_b = Variable(input_cropped3_b, requires_grad=False)
                        input_cropped2_b = input_cropped2_b.to(device)
                        input_cropped3_b = input_cropped3_b.to(device)
                        input_cropped_b = [input_cropped1_b, input_cropped2_b, input_cropped3_b]
                        #
                        # point_netG.eval()
                        netG_a2b.eval()
                        # fake_center1,fake_center2,fake  =point_netG(input_cropped)
                        fake_center1, fake_center2, fake = netG_a2b(input_cropped)
                        CD_loss = criterion_PointLoss(torch.squeeze(fake, 1), torch.squeeze(real_center, 1))
                        # YH
                        netG_b2a.eval()
                        fake_center1_b, fake_center2_b, fake_b = netG_b2a(input_cropped_b)
                        CD_loss_b = criterion_PointLoss(torch.squeeze(fake_b, 1), torch.squeeze(real_center_b, 1))
                        #
                        # print('test result:',CD_loss)
                        print('test result - CD_loss:', CD_loss)  # YH
                        print('test result - CD_loss_b:', CD_loss_b)  # YH
                        print('test result:', CD_loss + CD_loss_b)  # YH
                        # f.write('\n'+'test result:  %.4f'%(CD_loss))
                        # f.write('\n' + 'test result: CD_loss:%.4f CD_loss_b:%.4f SUM:%.4f' % (
                        # CD_loss, CD_loss_b, CD_loss + CD_loss_b))  # YH
                        break
                tic_toc.toc()  # 종료시간
                processing_time = tic_toc.tocvalue()
                ETA = (opt.niter - epoch) * processing_time * len(dataloader) + (len(dataloader) - i) * processing_time
                ETA_min = ETA / 60
                ETA_hour = ETA_min / 60
                print('procseeing time: %.4fsec ETA: %.4fhour' % (processing_time, ETA_hour))
                # f.write('\n' + 'processing time: %.4f' % (processing_time))
                # f.write('\n' + 'ETA: %.4f' % (ETA_hour))
                # f.close()
            schedulerD.step()
            schedulerG.step()
            # if epoch% 10 == 0:
            if epoch % 1 == 0:
                # torch.save({'epoch':epoch+1,
                #             'state_dict':point_netG.state_dict()},
                #             'Trained_Model/point_netG'+str(epoch)+'.pth' )
                # torch.save({'epoch':epoch+1,
                #             'state_dict':point_netD.state_dict()},
                #             'Trained_Model/point_netD'+str(epoch)+'.pth' )
                save_path = "./Trained_Model"
                if not os.path.isdir(save_path):
                    os.mkdir(save_path)
                # torch.save({'epoch': epoch + 1,
                #             'state_dict': point_netG.state_dict()},
                #            'Trained_Model/point_netG' + str(epoch) + '.pth')
                # torch.save({'epoch': epoch + 1,
                #             'state_dict': point_netD.state_dict()},
                #            'Trained_Model/point_netD' + str(epoch) + '.pth')
                # YH
                torch.save({'epoch': epoch + 1,
                            'state_dict': netG_a2b.state_dict()},
                           'Trained_Model/netG_a2b' + str(epoch) + '.pth')
                torch.save({'epoch': epoch + 1,
                            'state_dict': netD_a.state_dict()},
                           'Trained_Model/netD_a' + str(epoch) + '.pth')
                torch.save({'epoch': epoch + 1,
                            'state_dict': netG_b2a.state_dict()},
                           'Trained_Model/netG_b2a' + str(epoch) + '.pth')
                torch.save({'epoch': epoch + 1,
                            'state_dict': netD_b.state_dict()},
                           'Trained_Model/netD_b' + str(epoch) + '.pth')
                #

    #
    #############################
    ## ONLY G-NET
    ############################
    else:
        for epoch in range(resume_epoch, opt.niter):
            if epoch < 30:
                alpha1 = 0.01
                alpha2 = 0.02
            elif epoch < 80:
                alpha1 = 0.05
                alpha2 = 0.1
            else:
                alpha1 = 0.1
                alpha2 = 0.2

            for i, data in enumerate(dataloader, 0):

                real_point, target = data

                batch_size = real_point.size()[0]
                real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)
                input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)
                input_cropped1 = input_cropped1.data.copy_(real_point)
                real_point = torch.unsqueeze(real_point, 1)
                input_cropped1 = torch.unsqueeze(input_cropped1, 1)
                p_origin = [0, 0, 0]
                if opt.cropmethod == 'random_center':
                    choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]),
                              torch.Tensor([-1, 0, 0]), torch.Tensor([-1, 1, 0])]
                    for m in range(batch_size):
                        index = random.sample(choice, 1)
                        distance_list = []
                        p_center = index[0]
                        for n in range(opt.pnum):
                            distance_list.append(distance_squre(real_point[m, 0, n], p_center))
                        distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])

                        for sp in range(opt.crop_point_num):
                            input_cropped1.data[m, 0, distance_order[sp][0]] = torch.FloatTensor([0, 0, 0])
                            real_center.data[m, 0, sp] = real_point[m, 0, distance_order[sp][0]]
                real_point = real_point.to(device)
                real_center = real_center.to(device)
                input_cropped1 = input_cropped1.to(device)
                ############################
                # (1) data prepare
                ###########################
                real_center = Variable(real_center, requires_grad=True)
                real_center = torch.squeeze(real_center, 1)
                real_center_key1_idx = utils.farthest_point_sample(real_center, 64, RAN=False)
                real_center_key1 = utils.index_points(real_center, real_center_key1_idx)
                real_center_key1 = Variable(real_center_key1, requires_grad=True)

                real_center_key2_idx = utils.farthest_point_sample(real_center, 128, RAN=True)
                real_center_key2 = utils.index_points(real_center, real_center_key2_idx)
                real_center_key2 = Variable(real_center_key2, requires_grad=True)

                input_cropped1 = torch.squeeze(input_cropped1, 1)
                input_cropped2_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[1], RAN=True)
                input_cropped2 = utils.index_points(input_cropped1, input_cropped2_idx)
                input_cropped3_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[2], RAN=False)
                input_cropped3 = utils.index_points(input_cropped1, input_cropped3_idx)
                input_cropped1 = Variable(input_cropped1, requires_grad=True)
                input_cropped2 = Variable(input_cropped2, requires_grad=True)
                input_cropped3 = Variable(input_cropped3, requires_grad=True)
                input_cropped2 = input_cropped2.to(device)
                input_cropped3 = input_cropped3.to(device)
                input_cropped = [input_cropped1, input_cropped2, input_cropped3]
                point_netG = point_netG.train()
                point_netG.zero_grad()
                fake_center1, fake_center2, fake = point_netG(input_cropped)
                fake = torch.unsqueeze(fake, 1)
                ############################
                # (3) Update G network: maximize log(D(G(z)))
                ###########################

                CD_LOSS = criterion_PointLoss(torch.squeeze(fake, 1), torch.squeeze(real_center, 1))

                errG_l2 = criterion_PointLoss(torch.squeeze(fake, 1), torch.squeeze(real_center, 1)) \
                          + alpha1 * criterion_PointLoss(fake_center1, real_center_key1) \
                          + alpha2 * criterion_PointLoss(fake_center2, real_center_key2)

                errG_l2.backward()
                optimizerG.step()
                print('[%d/%d][%d/%d] Loss_G: %.4f / %.4f '
                      % (epoch, opt.niter, i, len(dataloader),
                         errG_l2, CD_LOSS))
                f = open('loss_PFNet.txt', 'a')
                f.write('\n' + '[%d/%d][%d/%d] Loss_G: %.4f / %.4f '
                        % (epoch, opt.niter, i, len(dataloader),
                           errG_l2, CD_LOSS))
                f.close()
            schedulerD.step()
            schedulerG.step()

            if epoch % 10 == 0:
                torch.save({'epoch': epoch + 1,
                            'state_dict': point_netG.state_dict()},
                           'Checkpoint/point_netG' + str(epoch) + '.pth')




