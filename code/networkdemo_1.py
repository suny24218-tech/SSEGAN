import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# =========================
# 3D 通道注意力模块（SE-like）
# 说明：轻量级仅在通道维做重新标定，不改变空间尺寸。
#      将其插入到 Conv_block3D 的末端以强化对SEA有用通道的响应（如边界/骨架特征）。
# =========================
class Attention3D(nn.Module):
    """
    3D 通道注意力（SE-like）。轻量级、低开销。
    reduction: 压缩比例（通常 8 或 16），当 channel 小时会退到 1。
    use_maxpool: 同时使用 GAP + GMP 的融合（对某些任务有利）。
    """
    def __init__(self, channel, reduction=8, use_maxpool=False):
        super(Attention3D, self).__init__()
        self.use_maxpool = use_maxpool
        # hidden 通道至少为 1，避免 channel < reduction 的情况
        hidden = max(1, channel // reduction)
        # 通过 1x1x1 卷积实现 FC 层（等价于全连接但保留 3D 形状）
        self.fc = nn.Sequential(
            nn.Conv3d(channel, hidden, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(hidden, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        if self.use_maxpool:
            # 可选的 max-pool 分支（与 avg-pool 融合）
            self.fc_max = nn.Sequential(
                nn.Conv3d(channel, hidden, kernel_size=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(hidden, channel, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
        # 全局平均池化，输出 [B, C, 1,1,1]
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        if self.use_maxpool:
            self.max_pool = nn.AdaptiveMaxPool3d(1)

    def forward(self, x):
        # x: [B, C, D, H, W]
        # 通过 GAP 获得通道描述
        a = self.avg_pool(x)    # [B, C, 1,1,1]
        out = self.fc(a)        # [B, C, 1,1,1] -> 每通道的缩放系数 (0..1)
        if self.use_maxpool:
            m = self.max_pool(x)
            out2 = self.fc_max(m)
            out = (out + out2) / 2.0
        # 广播相乘：返回 [B, C, D, H, W]，维度安全，不会引发形状不匹配
        return x * out

# weight init
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# =========================
# 基本卷积块：Conv_block3D
# 说明：在每个 block 的末端加入 Attention3D（通道注意力），
#       以便对 block 输出按通道自适应放缩，帮助突出有用特征并抑制伪影。
# =========================
class Conv_block3D(nn.Module):
    def __init__(self, n_ch_in, n_ch_out, m=0.1, use_att=True):
        """
        n_ch_in: 输入通道数
        n_ch_out: 输出通道数
        m: BatchNorm 的 momentum
        use_att: 是否启用 Attention3D（保留为 True 可快速开关）
        """
        super(Conv_block3D, self).__init__()
        self.conv1 = nn.Conv3d(n_ch_in, n_ch_out, 3, padding=0, bias=True)
        self.bn1 = nn.BatchNorm3d(n_ch_out, momentum=m)
        self.conv2 = nn.Conv3d(n_ch_out, n_ch_out, 3, padding=0, bias=True)
        self.bn2 = nn.BatchNorm3d(n_ch_out, momentum=m)
        self.conv3 = nn.Conv3d(n_ch_out, n_ch_out, 1, padding=0, bias=True)
        self.bn3 = nn.BatchNorm3d(n_ch_out, momentum=m)

        # 在 block 末端创建 Attention3D（通道注意力）
        # 使用默认 reduction=8（可根据 n_ch_out 调整）
        self.use_att = use_att
        if self.use_att:
            self.att = Attention3D(n_ch_out, reduction=8, use_maxpool=False)

    def forward(self, x):
        # x: [B, C_in, D, H, W]
        # conv1 -> bn1 -> act
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        # conv2 -> bn2 -> act
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        # conv3 -> bn3 -> act
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        # 在 block 末端应用通道注意力（不会改变形状）
        if self.use_att:
            x = self.att(x)
        return x


# =========================
# 上采样模块 Up3D（在 decoder 上采样后可选择在此处也插入注意力）
# 目前保持原样（若需可在此处也插入 Attention3D）
# =========================
class Up3D(nn.Module):
    def __init__(self, n_ch):
        super(Up3D, self).__init__()
        self.cov_1 = nn.Conv3d(n_ch, n_ch, 3, 1, 0)
        self.ac = nn.LeakyReLU(0.2, True)
        self.bn1 = nn.BatchNorm3d(n_ch)
        # 如果你想在上采样后再做一次通道注意力，可以在这里添加：
        # self.att_up = Attention3D(n_ch, reduction=8)

    def forward(self, x):
        # 先插值上采样，再做 conv + bn + act
        x = self.ac(self.bn1(self.cov_1(F.interpolate(x, scale_factor=2, mode='nearest'))))
        # 若使用 att_up，则应用： x = self.att_up(x)
        return x


# =========================
# 生成器结构 test_Generator3（保留原结构、只替换了 block 内部以加入 attention）
# =========================
class test_Generator3(nn.Module):
    def __init__(self, in_channel, step_channel=8):
        super(test_Generator3, self).__init__()
        self.in_channel = in_channel

        # 注意：Conv_block3D 默认已含 Attention3D（可在初始化时传入 use_att=False 以关闭）
        self.body1_1 = Conv_block3D(in_channel, step_channel)
        self.body1_2 = Conv_block3D(step_channel * 2, step_channel * 2)
        self.body1_3 = Conv_block3D(step_channel * 3, step_channel * 3)
        self.body1_4 = Conv_block3D(step_channel * 4, step_channel * 4)

        self.body2_1 = Conv_block3D(in_channel, step_channel)
        self.body3_1 = Conv_block3D(in_channel, step_channel)
        self.body4_1 = Conv_block3D(in_channel, step_channel)

        self.upbn1 = Up3D(step_channel)
        self.upbn2 = Up3D(2 * step_channel)
        self.upbn3 = Up3D(3 * step_channel)

        self.last_covs = nn.Sequential(
            nn.Conv3d(step_channel * 4, step_channel * 2, 5, 1, 0, bias=True),
            nn.BatchNorm3d(step_channel * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(step_channel * 2, step_channel, 3, 1, 0, bias=True),
            nn.BatchNorm3d(step_channel),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(step_channel, in_channel, 1, 1, 0, bias=True),
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x 是一个 list：x[0], x[1], x[2], x[3] 表示多尺度噪声输入
        # 这些输入应与原仓库生成器调用保持一致（shape 示例见 generate3D.py）
        out_1 = self.body1_1(x[0])        # 处理最小尺度输入
        out_1 = self.upbn1(out_1)         # 上采样一倍
        out_2_1 = self.body2_1(x[1])      # 处理第二尺度输入

        # 将上采样结果与第二尺度特征按通道拼接（dim=1）
        out_1 = torch.cat([out_1, out_2_1], dim=1)

        out_1 = self.body1_2(out_1)
        out_1 = self.upbn2(out_1)
        out_3_1 = self.body3_1(x[2])
        out_1 = torch.cat([out_1, out_3_1], dim=1)

        out_1 = self.body1_3(out_1)
        out_1 = self.upbn3(out_1)
        out_4_1 = self.body4_1(x[3])
        out_1 = torch.cat([out_1, out_4_1], dim=1)

        out_1 = self.body1_4(out_1)

        out_1 = self.last_covs(out_1)
        out_1 = self.tanh(out_1)

        return out_1


# =========================
# 判别器 VD（保持不变）
# =========================
class VD(nn.Module):
    def __init__(self, in_channel, pad):
        super(VD, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=pad)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=pad)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=pad)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=pad)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=pad)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=pad)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=pad)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=pad)

        self.pool1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.pool3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0)

        self.cov = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, inputs):
        out = self.relu(self.conv1_1(inputs))
        out = self.relu(self.conv1_2(out))
        out = self.relu(self.pool1(out))

        out = self.relu(self.conv2_1(out))
        out = self.relu(self.conv2_2(out))
        out = self.relu(self.pool2(out))

        out = self.relu(self.conv3_1(out))
        out = self.relu(self.conv3_2(out))
        out = self.relu(self.conv3_3(out))
        out = self.relu(self.conv3_4(out))
        out = self.relu(self.pool3(out))

        out = self.cov(out)
        return out