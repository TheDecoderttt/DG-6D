import os
import sys
import argparse
import numpy as np
import cv2
import torch
import pickle
from PIL import Image
import open3d as o3d

# 添加必要的路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.join(BASE_DIR, 'provider'),
    os.path.join(BASE_DIR, 'model'),
    os.path.join(BASE_DIR, 'utils'),
    os.path.join(BASE_DIR, 'lib'),
    os.path.join(BASE_DIR, 'lib', 'sphericalmap_utils'),
    os.path.join(BASE_DIR, 'lib', 'pointnet2')
])

from Net import Net
import gorilla


def fill_missing(    #点云填充
        dpt, cam_scale, scale_2_80m, fill_type='multiscale',
        extrapolate=False, show_process=False, blur_type='bilateral'
):
    dpt = dpt / cam_scale * scale_2_80m
    projected_depth = dpt.copy()
    if fill_type == 'fast':
        final_dpt = fill_in_fast(
            projected_depth, extrapolate=extrapolate, blur_type=blur_type,
            # max_depth=2.0
        )
    elif fill_type == 'multiscale':
        final_dpt, process_dict = fill_in_multiscale(
            projected_depth, extrapolate=extrapolate, blur_type=blur_type,
            show_process=show_process,
            max_depth=3.0
        )
    else:
        raise ValueError('Invalid fill_type {}'.format(fill_type))
    dpt = final_dpt / scale_2_80m * cam_scale
    return dpt

def load_model(config_path, checkpoint_path):
    """加载训练好的模型"""
    cfg = gorilla.Config.fromfile(config_path)
    model = Net(cfg.pose_net).cuda()
    gorilla.solver.load_checkpoint(model=model, filename=checkpoint_path)
    model.eval()
    return model, cfg

def load_data(data_dir, sample_num=1024):
    """加载 HouseCat6D 数据（RGB + 点云 + PKL 标签）"""
    # 1. 加载 RGB 图像
    rgb_path = os.path.join(data_dir, "rgb.png")
    rgb = Image.open(rgb_path).convert('RGB')
    rgb = np.array(rgb).astype(np.float32) / 255.0  # [H, W, 3]
    rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).cuda()  # [1, 3, H, W]

    # 2. 加载点云（.ply 或 .pcd）
    pcd_path = os.path.join(data_dir, "pointcloud.ply")
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)  # [N, 3]

    # 3. 随机采样固定数量的点
    if len(points) > sample_num:
        indices = np.random.choice(len(points), sample_num, replace=False)
        points = points[indices]
    pts = torch.from_numpy(points).float().unsqueeze(0).cuda()  # [1, N, 3]

    # 4. 生成 choose（索引，这里简单用 0~N-1）
    choose = torch.arange(sample_num).unsqueeze(0).cuda()  # [1, N]

    # 5. 加载 PKL 标签文件
    label_path = os.path.join(data_dir, "label.pkl")
    with open(label_path, 'rb') as f:
        label = pickle.load(f)  # 假设 PKL 文件包含字典，如 {'category_id': 1, 'pose': ...}

    # 6. 从标签中提取类别 ID
    cls_id = label['category_id']  # 假设 PKL 中有 'category_id' 字段
    cls = torch.tensor([cls_id], dtype=torch.long).cuda()  # [1]

    # 7. 打包成模型输入字典
    inputs = {
        'rgb': rgb,
        'pts': pts,
        'choose': choose,
        'category_label': cls
    }

    return inputs, points, label

def inference_and_visualize(model, data_dir, cfg):
    """执行推理并可视化结果（含 PKL 标签）"""
    # 1. 加载数据
    inputs, raw_points, label = load_data(data_dir, cfg.test_dataset.sample_num)

    # 2. 推理
    with torch.no_grad():
        pred_r, pred_t, pred_c = model(inputs)

    # 3. 解析结果
    R = pred_r.cpu().numpy().squeeze()  # [3, 3]
    t = pred_t.cpu().numpy().squeeze()  # [3]
    conf = pred_c.cpu().numpy().item()  # float

    print("Predicted Rotation (3x3):")
    print(R)
    print("\nPredicted Translation (XYZ):")
    print(t)
    print(f"\nConfidence: {conf:.4f}")

    # 4. 可视化（可选：对比预测与真实位姿）
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(raw_points)

    # 预测的坐标系
    pred_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    pred_frame.rotate(R, center=(0, 0, 0))
    pred_frame.translate(t)

    # 真实坐标系（如果 PKL 中有真实位姿）
    if 'pose' in label:
        gt_R = label['pose'][:3, :3]  # 假设 PKL 中存储了 4x4 位姿矩阵
        gt_t = label['pose'][:3, 3]
        gt_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, color=[0, 1, 0])  # 绿色
        gt_frame.rotate(gt_R, center=(0, 0, 0))
        gt_frame.translate(gt_t)
        o3d.visualization.draw_geometries([pcd, pred_frame, gt_frame])  # 显示预测和真值
    else:
        o3d.visualization.draw_geometries([pcd, pred_frame])  # 仅显示预测

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/HouseCat6D/housecat6d.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to HouseCat6D sample (with rgb.png, pointcloud.ply, label.pkl)")
    args = parser.parse_args()

    # 加载模型
    model, cfg = load_model(args.config, args.checkpoint)

    # 执行推理 + 可视化
    inference_and_visualize(model, args.data_dir, cfg)