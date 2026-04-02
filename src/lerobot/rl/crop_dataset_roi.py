"""
数据集 ROI 裁剪工具模块 (Dataset ROI Cropping Tool Module)
=====================================================

本模块提供了从 LeRobot 数据集中裁剪感兴趣区域 (ROI) 的工具。

主要函数:
    - select_rect_roi: 允许用户在图像上绘制矩形 ROI
    - select_square_roi_for_images: 为多张图像选择 ROI
    - convert_lerobot_dataset_to_cropped_lerobot_dataset: 转换数据集为裁剪版本

用法:
    python crop_dataset_roi.py --repo-id <数据集ID> --task <任务描述>
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as F  # type: ignore  # noqa: N812
from tqdm import tqdm  # type: ignore

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import DONE, REWARD


def select_rect_roi(img):
    """
    允许用户在图像上绘制矩形 ROI。

    用户必须点击并拖动来绘制矩形。
    - 拖动时，矩形会动态绘制。
    - 释放鼠标按钮时，矩形会固定。
    - 按 'c' 确认选择。
    - 按 'r' 重置选择。
    - 按 ESC 取消。

    返回:
        一个表示矩形 ROI 的元组 (top, left, height, width)，
        或者如果未选择有效 ROI 则返回 None。
    """
    # 创建图像的工作副本
    clone = img.copy()
    working_img = clone.copy()

    roi = None  # 将存储最终 ROI 为 (top, left, height, width)
    drawing = False
    index_x, index_y = -1, -1  # 初始点击坐标

    def mouse_callback(event, x, y, flags, param):
        nonlocal index_x, index_y, drawing, roi, working_img

        if event == cv2.EVENT_LBUTTONDOWN:
            # 开始绘制: 记录起始坐标
            drawing = True
            index_x, index_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                # 无论拖动方向如何，计算左上角和右下角
                top = min(index_y, y)
                left = min(index_x, x)
                bottom = max(index_y, y)
                right = max(index_x, x)
                # 显示绘制当前矩形的临时图像
                temp = working_img.copy()
                cv2.rectangle(temp, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.imshow("Select ROI", temp)

        elif event == cv2.EVENT_LBUTTONUP:
            # 完成绘制
            drawing = False
            top = min(index_y, y)
            left = min(index_x, x)
            bottom = max(index_y, y)
            right = max(index_x, x)
            height = bottom - top
            width = right - left
            roi = (top, left, height, width)  # (top, left, height, width)
            # 在工作图像上绘制最终矩形并显示
            working_img = clone.copy()
            cv2.rectangle(working_img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.imshow("Select ROI", working_img)

    # 创建窗口并设置回调
    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", mouse_callback)
    cv2.imshow("Select ROI", working_img)

    print("ROI 选择说明:")
    print("  - 点击并拖动绘制矩形 ROI。")
    print("  - 按 'c' 确认选择。")
    print("  - 按 'r' 重置并重新绘制。")
    print("  - 按 ESC 取消选择。")

    # 等待用户使用 'c' 确认、'r' 重置或 ESC 取消
    while True:
        key = cv2.waitKey(1) & 0xFF
        # 如果已绘制则确认 ROI
        if key == ord("c") and roi is not None:
            break
        # 重置: 清除 ROI 并恢复原始图像
        elif key == ord("r"):
            working_img = clone.copy()
            roi = None
            cv2.imshow("Select ROI", working_img)
        # 取消此图像的选择
        elif key == 27:  # ESC 键
            roi = None
            break

    cv2.destroyWindow("Select ROI")
    return roi


def select_square_roi_for_images(images: dict) -> dict:
    """
    对于提供的字典中的每个图像，打开一个窗口允许用户
    选择一个矩形 ROI。返回一个将每个键映射到元组
    (top, left, height, width) 表示 ROI 的字典。

    参数:
        images (dict): 字典，其中键是标识符，值是 OpenCV 图像。

    返回:
        dict: 将图像键映射到选择的矩形 ROI 的字典。
    """
    selected_rois = {}

    for key, img in images.items():
        if img is None:
            print(f"键 '{key}' 的图像为 None，跳过。")
            continue

        print(f"\n为键为 '{key}' 的图像选择矩形 ROI")
        roi = select_rect_roi(img)

        if roi is None:
            print(f"'{key}' 未选择有效 ROI。")
        else:
            selected_rois[key] = roi
            print(f"'{key}' 的 ROI: {roi}")

    return selected_rois


def get_image_from_lerobot_dataset(dataset: LeRobotDataset):
    """
    在数据集中找到第一行并提取图像以用于裁剪。
    """
    row = dataset[0]
    image_dict = {}
    for k in row:
        if "image" in k:
            image_dict[k] = deepcopy(row[k])
    return image_dict


def convert_lerobot_dataset_to_cropped_lerobot_dataset(
    original_dataset: LeRobotDataset,
    crop_params_dict: dict[str, tuple[int, int, int, int]],
    new_repo_id: str,
    new_dataset_root: str,
    resize_size: tuple[int, int] = (128, 128),
    push_to_hub: bool = False,
    task: str = "",
) -> LeRobotDataset:
    """
    通过遍历现有 LeRobotDataset 的回合和帧，对图像观测应用裁剪和调整大小，
    并保存具有转换数据的新数据集来转换它。

    参数:
        original_dataset (LeRobotDataset): 源数据集。
        crop_params_dict (Dict[str, Tuple[int, int, int, int]]):
            将观测键映射到裁剪参数 (top, left, height, width) 的字典。
        new_repo_id (str): 新数据集的仓库 ID。
        new_dataset_root (str): 将写入新数据集的根目录。
        resize_size (Tuple[int, int], optional): 裁剪后的目标大小 (height, width)。
            默认为 (128, 128)。

    返回:
        LeRobotDataset: 一个新的 LeRobotDataset，其中指定的图像观测已被裁剪和调整大小。
    """
    # 1. 创建一个用于写入的新（空）LeRobotDataset。
    new_dataset = LeRobotDataset.create(
        repo_id=new_repo_id,
        fps=int(original_dataset.fps),
        root=new_dataset_root,
        robot_type=original_dataset.meta.robot_type,
        features=original_dataset.meta.info["features"],
        use_videos=len(original_dataset.meta.video_keys) > 0,
    )

    # 更新每个将被裁剪的图像键的元数据:
    # (这里我们简单地将形状设置为最终 resize_size。)
    for key in crop_params_dict:
        if key in new_dataset.meta.info["features"]:
            new_dataset.meta.info["features"][key]["shape"] = [3] + list(resize_size)

    # TODO: 直接修改 mp4 视频 + meta info 特征，而不是重新创建数据集
    prev_episode_index = 0
    for frame_idx in tqdm(range(len(original_dataset))):
        frame = original_dataset[frame_idx]

        # 创建要添加到新数据集的帧副本
        new_frame = {}
        for key, value in frame.items():
            if key in ("task_index", "timestamp", "episode_index", "frame_index", "index", "task"):
                continue
            if key in (DONE, REWARD):
                # if not isinstance(value, str) and len(value.shape) == 0:
                value = value.unsqueeze(0)

            if key in crop_params_dict:
                top, left, height, width = crop_params_dict[key]
                # 应用裁剪然后调整大小。
                cropped = F.crop(value, top, left, height, width)
                value = F.resize(cropped, resize_size)
                value = value.clamp(0, 1)
            if key.startswith("complementary_info") and isinstance(value, torch.Tensor) and value.dim() == 0:
                value = value.unsqueeze(0)
            new_frame[key] = value

        new_frame["task"] = task
        new_dataset.add_frame(new_frame)

        if frame["episode_index"].item() != prev_episode_index:
            # 保存回合
            new_dataset.save_episode()
            prev_episode_index = frame["episode_index"].item()

    # 保存最后一个回合
    new_dataset.save_episode()

    if push_to_hub:
        new_dataset.push_to_hub()

    return new_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop rectangular ROIs from a LeRobot dataset.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="lerobot",
        help="The repository id of the LeRobot dataset to process.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="The root directory of the LeRobot dataset.",
    )
    parser.add_argument(
        "--crop-params-path",
        type=str,
        default=None,
        help="The path to the JSON file containing the ROIs.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Whether to push the new dataset to the hub.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="The natural language task to describe the dataset.",
    )
    parser.add_argument(
        "--new-repo-id",
        type=str,
        default=None,
        help="The repository id for the new cropped and resized dataset. If not provided, it defaults to `repo_id` + '_cropped_resized'.",
    )
    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)

    images = get_image_from_lerobot_dataset(dataset)
    images = {k: v.cpu().permute(1, 2, 0).numpy() for k, v in images.items()}
    images = {k: (v * 255).astype("uint8") for k, v in images.items()}

    if args.crop_params_path is None:
        rois = select_square_roi_for_images(images)
    else:
        with open(args.crop_params_path) as f:
            rois = json.load(f)

    # 打印选择的矩形 ROI
    print("\n选择的矩形感兴趣区域 (top, left, height, width):")
    for key, roi in rois.items():
        print(f"{key}: {roi}")

    new_repo_id = args.new_repo_id if args.new_repo_id else args.repo_id + "_cropped_resized"

    if args.new_repo_id:
        new_dataset_name = args.new_repo_id.split("/")[-1]
        # 父目录 1: HF 用户, 父目录 2: HF LeRobot 主页
        new_dataset_root = dataset.root.parent.parent / new_dataset_name
    else:
        new_dataset_root = Path(str(dataset.root) + "_cropped_resized")

    cropped_resized_dataset = convert_lerobot_dataset_to_cropped_lerobot_dataset(
        original_dataset=dataset,
        crop_params_dict=rois,
        new_repo_id=new_repo_id,
        new_dataset_root=new_dataset_root,
        resize_size=(128, 128),
        push_to_hub=args.push_to_hub,
        task=args.task,
    )

    meta_dir = new_dataset_root / "meta"
    meta_dir.mkdir(exist_ok=True)

    with open(meta_dir / "crop_params.json", "w") as f:
        json.dump(rois, f, indent=4)
