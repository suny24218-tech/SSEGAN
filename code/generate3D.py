import torch
import os
from functiondemo import *
from networkdemo import *
import numpy as np
os.chdir("/root/autodl-tmp/3D")
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# 定义反归一化函数：将范围在(-1, 1)的图像数据转换为(0, 1)
def de_norm(image):  # (-1,1)->(0,1)
    out = (image + 1) / 2  # 反归一化计算：(x + 1)/2
    return out.clamp(0, 1)  # 限制输出在[0,1]范围内，避免溢出

# 定义保存3D模型切片的函数：从3D数据中提取指定维度的切片并保存为图像
def cubeslice(x, dim, path, h, w):
    hwd = x.shape[2:5]  # 获取3D数据的空间维度（h, w, d）
    for i in range(hwd[dim]):  # 遍历指定维度的所有切片
        slice = x[0, :, :, :, i]  # 提取第i个切片（x为批量数据，取第0个样本）
        slice = slice.transpose(1, 2, 0)  # 调整维度顺序：(通道, 高, 宽) -> (高, 宽, 通道)
        img = Image.fromarray(np.uint8(slice))  # 将数组转换为PIL图像（数据类型为uint8）
        img = img.convert('L')  # 转换为灰度图像（L模式）
        img = img.convert('RGB')  # 再转换为RGB模式（可能为了统一输出格式）
        img.save(path + '/' + str(h*w+i).zfill(4) + '.bmp')  # 保存图像到指定路径，文件名格式为"h*w+i"（补零至4位）


if __name__ == '__main__':
    h = 400  # 生成的3D图像在某维度的尺寸（需为8的倍数，与网络下采样/上采样兼容）
    w = 8  # 每次生成的切片厚度（需为8的倍数，与网络结构兼容）
    
    # 定义输出目录路径
    path1 = './output/Test2'

    if os.path.exists(path1):
        print('Folder already exists')
    else:
        os.mkdir(path1)
        print('Create a folder')

    # 预训练生成器模型的路径
    path_g = './model/Example.pth'
    net_G = test_Generator3(3, 8)  # 初始化生成器网络：输入通道数为3，步长通道数为8（与训练时一致）
    net_G = net_G.to(device)
    # checkpoint_G = torch.load(path_g, map_location='cuda:0')
    checkpoint_G = torch.load(path_g, map_location=device)   # 加载预训练模型权重（map_location确保权重加载到指定设备）
    net_G.load_state_dict(checkpoint_G['model'])  # 将权重加载到生成器网络中
    net_G.eval()  # 设置生成器为评估模式（关闭dropout等训练时特有的层）
    # 用于存储生成的3D数据（此处未直接存储，而是通过切片保存）
    solid = []
    # 初始化输入随机噪声张量：4个不同尺度的噪声（对应生成器的多尺度输入）
    # 尺寸计算考虑网络上采样需求，添加额外像素（10/14）以保证输出尺寸正确
    input_rand = [torch.randn([1, 3, int(h/8 + 10), int(h/8 + 10), int(w/8 + 10)], device=device),
                torch.randn([1, 3, int(h/4 + 14), int(h/4 + 14), int(w/4 + 14)], device=device),
                torch.randn([1, 3, int(h/2 + 14), int(h/2 + 14), int(w/2 + 14)], device=device),
                torch.randn([1, 3, int(h+14), int(h+14), int(w+14)], device=device)]

    # 分块生成3D模型：总次数为h/w（每次生成w厚度的切片，累计生成h厚度）
    for i in range(int(h/w)):
        # 生成当前块的输入噪声（与input_rand同尺度）
        test1 = [torch.randn([1, 3, int(h/8 + 10), int(h/8 + 10), int(w/8 + 10)], device=device),
                    torch.randn([1, 3, int(h/4 + 14), int(h/4 + 14), int(w/4 + 14)], device=device),
                    torch.randn([1, 3, int(h/2 + 14), int(h/2 + 14), int(w/2 + 14)], device=device),
                    torch.randn([1, 3, int(h+14), int(h+14), int(w+14)], device=device)]

        # 噪声拼接：将上一块噪声的末尾部分作为当前块噪声的开头，保证3D结构连续性
        test1[0][:,:,:,:,:10] = input_rand[0][:,:,:,:,-10:]
        test1[1][:,:,:,:,:14] = input_rand[1][:,:,:,:,-14:]
        test1[2][:,:,:,:,:14] = input_rand[2][:,:,:,:,-14:]
        test1[3][:,:,:,:,:14] = input_rand[3][:,:,:,:,-14:]

        # 更新输入噪声为当前块噪声，用于下一次循环拼接
        input_rand = test1
        # 关闭梯度计算（推理阶段无需更新权重，节省内存）
        with torch.no_grad():
            im = net_G(test1)  # 生成器输出当前块的3D数据（范围(-1,1)）
            im = de_norm(im).cpu().numpy()*255  # 反归一化并转换为CPU上的numpy数组，缩放至0-255
            cubeslice(im, -1, path1, i, w)  # 保存当前块的切片（dim=-1表示沿最后一个维度切片）

