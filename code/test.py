"""
test_grain_detection.py - 测试改进后的晶粒检测功能
验证晶粒统计是否正确（包括边界晶粒按0.5计算）
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from functiondemo_1 import extract_target_statistics, compute_grain_statistics
import os
os.chdir("/root/autodl-tmp/3D/CEM3DMG-main")
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def visualize_grain_detection(image_path, output_dir='./test_results'):
    """
    可视化晶粒检测结果

    参数:
        image_path: 输入图像路径
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_color = cv2.imread(image_path)

    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    print("=" * 60)
    print(f"测试图像: {image_path}")
    print("=" * 60)

    # 提取统计特征
    stats = compute_grain_statistics(image, threshold=127, min_grain_size=50, bins=50)

    # 打印详细统计信息
    print(f"\n检测结果:")
    print(f"  原始检测晶粒数: {stats['grain_count_raw']}")
    print(f"  封闭晶粒数: {stats['closed_grains']}")
    print(f"  边界晶粒数: {stats['boundary_grains']}")
    print(f"  加权晶粒数: {stats['grain_count']:.2f} (边界×0.5)")
    print(f"  平均面积: {stats['mean_area']:.2f} 像素")
    print(f"  标准差: {stats['std_area']:.2f}")
    print(f"  中位数面积: {stats['median_area']:.2f} 像素")

    # 在图像上标注晶粒
    annotated_image = image_color.copy()

    for info in stats['grain_info']:
        centroid = info['centroid']
        is_closed = info['is_closed']

        # 封闭晶粒用绿色，边界晶粒用红色
        color = (0, 255, 0) if is_closed else (255, 0, 0)

        # 绘制质心
        cv2.circle(annotated_image,
                   (int(centroid[1]), int(centroid[0])),
                   3, color, -1)

        # 可选：绘制边界框
        min_row, min_col, max_row, max_col = info['bbox']
        cv2.rectangle(annotated_image,
                      (min_col, min_row),
                      (max_col, max_row),
                      color, 1)

    # 添加图例
    legend_y = 30
    cv2.putText(annotated_image, "Green: Closed grains",
                (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)
    cv2.putText(annotated_image, "Red: Boundary grains (0.5x)",
                (10, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 0, 0), 2)

    # 保存标注图像
    output_path = os.path.join(output_dir, 'grain_detection_result.png')
    cv2.imwrite(output_path, annotated_image)
    print(f"\n标注图像已保存: {output_path}")

    # 绘制晶粒面积分布直方图
    plt.figure(figsize=(10, 6))

    # 分别绘制封闭和边界晶粒
    closed_areas = [info['area'] for info in stats['grain_info'] if info['is_closed']]
    boundary_areas = [info['area'] for info in stats['grain_info'] if not info['is_closed']]

    plt.hist(closed_areas, bins=30, alpha=0.7, label=f'封闭晶粒 (n={len(closed_areas)})', color='green')
    plt.hist(boundary_areas, bins=30, alpha=0.7, label=f'边界晶粒 (n={len(boundary_areas)})', color='red')

    plt.xlabel('晶粒面积 (像素)')
    plt.ylabel('数量')
    plt.title(f'晶粒面积分布\n总计: {stats["grain_count"]:.1f} 个晶粒 (边界按0.5计)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    hist_path = os.path.join(output_dir, 'grain_area_distribution.png')
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"面积分布图已保存: {hist_path}")

    # 绘制标记后的连通域
    labeled_viz = np.zeros_like(image_color)
    labeled_image = stats['labeled_image']

    # 为每个晶粒分配随机颜色
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(labeled_image.max() + 1, 3))
    colors[0] = [0, 0, 0]  # 背景为黑色

    for label in range(1, labeled_image.max() + 1):
        mask = labeled_image == label
        labeled_viz[mask] = colors[label]

    labeled_path = os.path.join(output_dir, 'labeled_grains.png')
    cv2.imwrite(labeled_path, labeled_viz)
    print(f"标记图像已保存: {labeled_path}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

    return stats


def compare_with_reference(image_path, reference_count, reference_mean, output_dir='./test_results'):
    """
    与参考值对比

    参数:
        image_path: 输入图像路径
        reference_count: 参考晶粒数量
        reference_mean: 参考平均尺寸
        output_dir: 输出目录
    """
    print("\n" + "=" * 60)
    print("与参考值对比")
    print("=" * 60)

    stats = extract_target_statistics(image_path, threshold=127, min_grain_size=50, bins=50)

    count_error = abs(stats['grain_count'] - reference_count) / reference_count * 100
    mean_error = abs(stats['mean_area'] - reference_mean) / reference_mean * 100

    print(f"\n对比结果:")
    print(f"  参考晶粒数: {reference_count}")
    print(f"  检测晶粒数: {stats['grain_count']:.2f}")
    print(f"  数量误差: {count_error:.2f}%")
    print(f"\n  参考平均面积: {reference_mean:.2f} 像素")
    print(f"  检测平均面积: {stats['mean_area']:.2f} 像素")
    print(f"  面积误差: {mean_error:.2f}%")

    if count_error < 10 and mean_error < 10:
        print("\n✓ 检测结果良好（误差<10%）")
    else:
        print("\n⚠ 检测结果存在较大误差")

    return stats


if __name__ == '__main__':
    # 测试图像路径（修改为你的图像路径）
    test_image = './image/hcg.png'

    # 检查文件是否存在
    if not os.path.exists(test_image):
        print(f"错误: 找不到图像文件 {test_image}")
        print("请修改 test_image 变量为正确的图像路径")
    else:
        # 执行可视化测试
        stats = visualize_grain_detection(test_image, output_dir='./test_results')

        # 如果你有参考值，可以进行对比
        # 例如：根据你提供的代码，如果封闭晶粒=k，边界晶粒=r，则总数约为 k + 0.5*r
        # reference_count = 42.5  # 示例值
        # reference_mean = 3500.0  # 示例值
        # compare_with_reference(test_image, reference_count, reference_mean)

        print("\n所有测试结果已保存到 ./test_results/ 目录")