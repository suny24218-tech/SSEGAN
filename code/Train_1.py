import torch
import torch.optim as optimizer
import os
import datetime
from functiondemo_1 import *
from networkdemo import *
from torchvision import transforms
import numpy as np
import argparse

os.chdir("/root/autodl-tmp/Anisotropy")

#
parse = argparse.ArgumentParser()

parse.add_argument('--input_dir', default='./image/', help='input image dir')
parse.add_argument('--train_dir', default='./trained/', help='slices in train process')
parse.add_argument('--model_dir', default='./model/', help='model parameter')
parse.add_argument('--check_dir', default='./checkpoint/', help='checkpoint model parameter')

# ========== 🔴 修改点1：添加第二个方向的图像参数 ==========
parse.add_argument('--img_xy', default='2.1x.png', help='XY direction 2D image')
parse.add_argument('--img_xz', default='2.1y.png', help='XZ direction 2D image')

parse.add_argument('--scale_N', default=5, type=int, help='learning scale. default=5')
parse.add_argument('--lr_g', default=5e-4, type=float, help='generator learning rate')
parse.add_argument('--lr_d', default=3e-4, type=float, help='discriminator learning rate')
parse.add_argument('--beta', type=float, default=0.9, help='beta1 for adam. default=0.9')
parse.add_argument('--gp_lambda', type=float, default=8, help='gradient penalty weight')
parse.add_argument('--gamma', type=float, help='scheduler gamma', default=0.9)
parse.add_argument('--batch_size', default=1, type=int, help='generator batch_size')
parse.add_argument('--batch_slice', default=16, type=int, help='slice batch_size')  # 原16
parse.add_argument('--direction_w', default=32, type=int, help='size of 3D image on training direction,multiples of 8')
parse.add_argument('--cuda', default='cuda:0', help='cuda number')
parse.add_argument('--pre', default=False, help='yes or no fix pre-training parameters')
parse.add_argument('--is_check', default=False, help='yes or no use previous checkpoint train')
# tscale and cha are not fixed, it needs to be set according to the size of the examples
parse.add_argument('--tscale', default=304, type=int, help='max crop size, multiples of 8')
parse.add_argument('--cha', default=32, type=int, help='interval of multi cropped size, multiples of 8')

parse.add_argument('--vggstep', default=50000, type=int, help='perceptual loss')
parse.add_argument('--vggk', default=0.01, type=float, help='the coefficient of perceptual loss')  #0.01

parse.add_argument('--iter_train', default=12000, type=int, help='total train iter, 50000-120000')
parse.add_argument('--iter_slice', default=100, type=int, help='every num save train slice')
parse.add_argument('--iter_check', default=2000, type=int, help='every num save checkpoint')
parse.add_argument('--iter_model', default=1000, type=int, help='every num save model')
parse.add_argument('--step_channel', default=8, type=int, help='step_channel')

opt = parse.parse_args()

# device
device = torch.device(opt.cuda)
batch_size = opt.batch_size

input_image_xy = opt.img_xy
input_image_xz = opt.img_xz
opt.input_dir_xy = opt.input_dir + input_image_xy
opt.input_dir_xz = opt.input_dir + input_image_xz
image_xy, image_xz = load_dual_images(opt.input_dir_xy, opt.input_dir_xz)

# image info
opt.channel = len(image_xy.split())
opt.real_img_h = image_xy.size[0]
opt.real_img_w = image_xy.size[1]
print(f"XY image info: channel={opt.channel}, H={opt.real_img_h}, W={opt.real_img_w}")
print(f"XZ image info: channel={len(image_xz.split())}, H={image_xz.size[0]}, W={image_xz.size[1]}")

# create trained dir
time_now = datetime.datetime.now()
opt.train_dir = opt.train_dir + time_now.strftime("%Y-%m-%d-%H-%M-%S-") + input_image_xy[0:input_image_xy.rfind('.')]
os.mkdir(opt.train_dir)

# create model dir
opt.model_dir = opt.model_dir + time_now.strftime("%Y-%m-%d-%H-%M-%S-") + input_image_xy[0:input_image_xy.rfind('.')]
os.mkdir(opt.model_dir)
# create check dir
opt.check_dir = opt.check_dir + time_now.strftime("%Y-%m-%d-%H-%M-%S-") + input_image_xy[0:input_image_xy.rfind('.')]
os.mkdir(opt.check_dir)

# load vgg19_model
vgg = VGG(pool='avg', pad=1)
vgg.load_state_dict(torch.load('./code/vgg_conv.pth'))
for param in vgg.parameters():
    param.requires_grad = False
vgg.to(device)

# Perceptual Loss
loss_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
loss_fns = [GramMSELoss()] * len(loss_layers)
loss_fns = [loss_fn.to(device) for loss_fn in loss_fns]
wd = [1, 1, 1, 1, 1]

# setup generator network
net_G = test_Generator3(opt.channel, opt.step_channel)
net_G.apply(weights_init)
net_G.to(device)
optimizer_G = optimizer.Adam(net_G.parameters(), lr=opt.lr_g, betas=(opt.beta, 0.999))


net_Ds = []  # 每个元素是一个列表，包含2个判别器 [D_xy, D_xz]
optimizer_Ds = []  # 每个元素是一个列表，包含2个优化器 [opt_xy, opt_xz]

for i in range(opt.scale_N):
    # XY方向判别器
    vgg_dict = torch.load('./code/vgg_conv.pth')
    net_D_xy = VD(in_channel=3, pad=1)
    net_D_xy.apply(weights_init)
    net_D_xy_dict = net_D_xy.state_dict()
    vgg_dict_xy = {k: v for k, v in vgg_dict.items() if k in net_D_xy_dict}
    net_D_xy_dict.update(vgg_dict_xy)
    net_D_xy.load_state_dict(net_D_xy_dict)
    for name, param in net_D_xy.named_parameters():
        if name.find('conv') != -1:
            param.requires_grad = opt.pre
    optimizer_D_xy = optimizer.Adam(net_D_xy.parameters(), lr=opt.lr_d, betas=(opt.beta, 0.999))
    net_D_xy.to(device)

    # XZ方向判别器
    vgg_dict = torch.load('./code/vgg_conv.pth')
    net_D_xz = VD(in_channel=3, pad=1)
    net_D_xz.apply(weights_init)
    net_D_xz_dict = net_D_xz.state_dict()
    vgg_dict_xz = {k: v for k, v in vgg_dict.items() if k in net_D_xz_dict}
    net_D_xz_dict.update(vgg_dict_xz)
    net_D_xz.load_state_dict(net_D_xz_dict)
    for name, param in net_D_xz.named_parameters():
        if name.find('conv') != -1:
            param.requires_grad = opt.pre
    optimizer_D_xz = optimizer.Adam(net_D_xz.parameters(), lr=opt.lr_d, betas=(opt.beta, 0.999))
    net_D_xz.to(device)

    net_Ds.append([net_D_xy, net_D_xz])
    optimizer_Ds.append([optimizer_D_xy, optimizer_D_xz])

# save train config
save_config(opt, opt.train_dir)

# train
if __name__ == '__main__':
    Gloss_history = np.zeros((opt.iter_train))
    # ========== 🔴 修改点4：修改损失历史记录，每个scale有2个方向 ==========
    Dloss_xy_history = np.zeros((opt.scale_N, opt.iter_train))
    Dloss_xz_history = np.zeros((opt.scale_N, opt.iter_train))
    Vloss_history = np.zeros((opt.iter_train))

    tt, t = opt.tscale, opt.cha
    h_rand = [tt - t * 4, tt - t * 3, tt - t * 2, tt - t, tt]
    time_now = datetime.datetime.now()
    strat = 0
    if opt.is_check == True:
        filepath = 'checkpoint/XXXX_checkpoint.pth'
        strat = load_checkpoint(filepath, net_G, net_Ds, optimizer_G, optimizer_Ds, device) + 1

    for iter_num in range(strat, opt.iter_train):
        if (iter_num + 1) % 5000 == 0:
            opt.gp_lambda = opt.gp_lambda * 0.9

        scale_choice = np.random.randint(0, opt.scale_N)
        h = h_rand[scale_choice]
        w = opt.direction_w
        fake_slices = []

        for d in range(2):  # 从3改为2
            if d == 0:  # XY方向（沿Z轴切）
                input_rand = [
                    torch.randn([opt.batch_size, 3, int(w / 8 + 10), int(h / 8 + 10), int(h / 8 + 10)], device=device),
                    torch.randn([opt.batch_size, 3, int(w / 4 + 14), int(h / 4 + 14), int(h / 4 + 14)], device=device),
                    torch.randn([opt.batch_size, 3, int(w / 2 + 14), int(h / 2 + 14), int(h / 2 + 14)], device=device),
                    torch.randn([opt.batch_size, 3, int(w + 14), int(h + 14), int(h + 14)], device=device)]
                fake = net_G(input_rand)
                fake = F.interpolate(fake, size=[w, h_rand[-3], h_rand[-3]], mode='trilinear')
                fake_slice = cutone_d0(fake, opt.batch_slice).to(device)
                fake_slices.append(fake_slice)

            if d == 1:  # XZ方向（沿Y轴切）
                input_rand = [
                    torch.randn([opt.batch_size, 3, int(h / 8 + 10), int(w / 8 + 10), int(h / 8 + 10)], device=device),
                    torch.randn([opt.batch_size, 3, int(h / 4 + 14), int(w / 4 + 14), int(h / 4 + 14)], device=device),
                    torch.randn([opt.batch_size, 3, int(h / 2 + 14), int(w / 2 + 14), int(h / 2 + 14)], device=device),
                    torch.randn([opt.batch_size, 3, int(h + 14), int(w + 14), int(h + 14)], device=device)]
                fake = net_G(input_rand)
                fake = F.interpolate(fake, size=[h_rand[-3], w, h_rand[-3]], mode='trilinear')
                fake_slice = cutone_d1(fake, opt.batch_slice).to(device)
                fake_slices.append(fake_slice)

        fake_slices = torch.cat(fake_slices, dim=0)

        optimizer_G.zero_grad()
        if (iter_num + 1) % 10 == 0 and (iter_num + 1) <= opt.vggstep:
            texture_xy = real_rand_cropopt_xy(image_xy, h, batch_size * opt.batch_slice, h_rand[-3], device)
            texture_xz = real_rand_cropopt_xz(image_xz, h, batch_size * opt.batch_slice, h_rand[-3], device)
            texture = torch.cat([texture_xy, texture_xz], dim=0)

            texture = (texture) * 255.
            targets = [GramMatrix()(f).detach() for f in vgg(texture, loss_layers)]
            fake_slices_gram = vgg(de_norm(fake_slices) * 255, loss_layers)
            vgg_loss = opt.vggk * sum([wd[a] * loss_fns[a](f, targets[a]) for a, f in enumerate(fake_slices_gram)])
            Vloss_history[iter_num] = Vloss_history[iter_num] + vgg_loss.item()
            vgg_loss.backward(retain_graph=True)

        optimizer_Ds[scale_choice][0].zero_grad()
        optimizer_Ds[scale_choice][1].zero_grad()
        optimizer_G.zero_grad()

        # 分别计算两个方向的判别器输出
        fake_xy = fake_slices[:opt.batch_slice]
        fake_xz = fake_slices[opt.batch_slice:]

        out_fake_xy = net_Ds[scale_choice][0](fake_xy)
        out_fake_xz = net_Ds[scale_choice][1](fake_xz)

        # 双方向GAN loss
        errG = -(out_fake_xy.mean() + out_fake_xz.mean()) / 2.0
        errG.backward()
        errorG = errG.data
        Gloss_history[iter_num] = Gloss_history[iter_num] + errorG.item()
        optimizer_G.step()

        del fake_slices, input_rand

        for s in range(opt.scale_N):
            h = h_rand[s]
            w = opt.direction_w
            fake_slices = []
            for d in range(2):  # 两个方向
                if d == 0:  # XY方向
                    input_rand = [torch.randn([opt.batch_size, 3, int(w / 8 + 10), int(h / 8 + 10), int(h / 8 + 10)],
                                              device=device),
                                  torch.randn([opt.batch_size, 3, int(w / 4 + 14), int(h / 4 + 14), int(h / 4 + 14)],
                                              device=device),
                                  torch.randn([opt.batch_size, 3, int(w / 2 + 14), int(h / 2 + 14), int(h / 2 + 14)],
                                              device=device),
                                  torch.randn([opt.batch_size, 3, int(w + 14), int(h + 14), int(h + 14)],
                                              device=device)]
                    fake = net_G(input_rand)
                    fake = F.interpolate(fake, size=[w, h_rand[-3], h_rand[-3]], mode='trilinear')
                    fake_slice = cutone_d0(fake, opt.batch_slice).to(device)
                    fake_slices.append(fake_slice)

                if d == 1:  # XZ方向
                    input_rand = [torch.randn([opt.batch_size, 3, int(h / 8 + 10), int(w / 8 + 10), int(h / 8 + 10)],
                                              device=device),
                                  torch.randn([opt.batch_size, 3, int(h / 4 + 14), int(w / 4 + 14), int(h / 4 + 14)],
                                              device=device),
                                  torch.randn([opt.batch_size, 3, int(h / 2 + 14), int(w / 2 + 14), int(h / 2 + 14)],
                                              device=device),
                                  torch.randn([opt.batch_size, 3, int(h + 14), int(w + 14), int(h + 14)],
                                              device=device)]
                    fake = net_G(input_rand)
                    fake = F.interpolate(fake, size=[h_rand[-3], w, h_rand[-3]], mode='trilinear')
                    fake_slice = cutone_d1(fake, opt.batch_slice).to(device)
                    fake_slices.append(fake_slice)

            fake_slices = torch.cat(fake_slices, dim=0)

            texture_xy = real_rand_cropopt_xy(image_xy, h, batch_size * opt.batch_slice, h_rand[-3], device)
            texture_xz = real_rand_cropopt_xz(image_xz, h, batch_size * opt.batch_slice, h_rand[-3], device)

            real_xy = norm(texture_xy).to(device)
            real_xz = norm(texture_xz).to(device)

            fake_xy = fake_slices[:opt.batch_slice]
            fake_xz = fake_slices[opt.batch_slice:]

            optimizer_Ds[s][0].zero_grad()
            real_out_xy = net_Ds[s][0](real_xy)
            D_real_xy = -real_out_xy.mean()
            D_real_xy.backward()

            fake_out_xy = net_Ds[s][0](fake_xy.detach())
            D_fake_xy = fake_out_xy.mean()
            D_fake_xy.backward()

            gp_xy = gradient_penalty(net_Ds[s][0], real_xy, fake_xy.detach(), opt.gp_lambda, device)
            gp_xy.backward()
            Dloss_xy_history[s][iter_num] = D_real_xy.item() + D_fake_xy.item() + gp_xy.item()
            optimizer_Ds[s][0].step()

            optimizer_Ds[s][1].zero_grad()
            real_out_xz = net_Ds[s][1](real_xz)
            D_real_xz = -real_out_xz.mean()
            D_real_xz.backward()

            fake_out_xz = net_Ds[s][1](fake_xz.detach())
            D_fake_xz = fake_out_xz.mean()
            D_fake_xz.backward()

            gp_xz = gradient_penalty(net_Ds[s][1], real_xz, fake_xz.detach(), opt.gp_lambda, device)
            gp_xz.backward()
            Dloss_xz_history[s][iter_num] = D_real_xz.item() + D_fake_xz.item() + gp_xz.item()
            optimizer_Ds[s][1].step()

            del input_rand, fake_slices

        if (iter_num + 1) % 10 == 0:
            print('iter:', iter_num,
                  ', D_XY0:', format(Dloss_xy_history[0][iter_num], '.3f'),
                  ', D_XZ0:', format(Dloss_xz_history[0][iter_num], '.3f'),
                  ', D_XY1:', format(Dloss_xy_history[1][iter_num], '.2f'),
                  ', D_XZ1:', format(Dloss_xz_history[1][iter_num], '.2f'),
                  ', G:', format(Gloss_history[iter_num], '.2f'),
                  ', VGG:', format(Vloss_history[iter_num], '.4f'))

        # train process show results
        if (iter_num + 1) % opt.iter_slice == 0:
            net_G.eval()
            with torch.no_grad():
                w = 8
                h = 400
                for dd in range(2):  # 只保存两个方向
                    if dd == 0:  # XY方向
                        a = [torch.randn([opt.batch_size, 3, int(w / 8 + 10), int(h / 8 + 10), int(h / 8 + 10)],
                                         device=device),
                             torch.randn([opt.batch_size, 3, int(w / 4 + 14), int(h / 4 + 14), int(h / 4 + 14)],
                                         device=device),
                             torch.randn([opt.batch_size, 3, int(w / 2 + 14), int(h / 2 + 14), int(h / 2 + 14)],
                                         device=device),
                             torch.randn([opt.batch_size, 3, int(w + 14), int(h + 14), int(h + 14)], device=device)]

                    if dd == 1:  # XZ方向
                        a = [torch.randn([opt.batch_size, 3, int(h / 8 + 10), int(w / 8 + 10), int(h / 8 + 10)],
                                         device=device),
                             torch.randn([opt.batch_size, 3, int(h / 4 + 14), int(w / 4 + 14), int(h / 4 + 14)],
                                         device=device),
                             torch.randn([opt.batch_size, 3, int(h / 2 + 14), int(w / 2 + 14), int(h / 2 + 14)],
                                         device=device),
                             torch.randn([opt.batch_size, 3, int(h + 14), int(w + 14), int(h + 14)], device=device)]

                    im = net_G(a)
                    im = de_norm(im)
                    test_slice = fake_rand_crop(im, 1, 1, dd, w, opt.cuda).to(device)
                    test_slice = test_slice.squeeze(0).cpu()
                    test_slice = toPIL(test_slice).convert('L').convert('RGB')
                    test_slice.save(
                        opt.train_dir + '/' + str(iter_num).zfill(
                            5) + '_' + 'd' + str(dd) + '.bmp')
                del a, im, test_slice
            net_G.train()

        if (iter_num + 1) % opt.iter_model == 0 and (iter_num + 1) >= 6000:
            checkpoint_G = {'model': net_G.state_dict(), 'optimizer': optimizer_G.state_dict(), 'epoch': iter_num}
            torch.save(checkpoint_G, opt.model_dir + '/' + str(iter_num) + '_' + 'g_model.pth')
        if (iter_num + 1) % opt.iter_check == 0:
            save_checkpoint(net_G, net_Ds, optimizer_G, optimizer_Ds, iter_num,
                            opt.check_dir + '/' + str(iter_num) + '_' + 'checkpoint.pth')