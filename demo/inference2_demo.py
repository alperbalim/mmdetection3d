from mmdet3d.datasets import KittiDataset
import torch
import numpy as np
from mmdet3d.apis import LidarDet3DInferencer
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes
from mmengine import load
from mmdet3d.datasets.transforms.loading import LoadAnnotations3D

#dict(points='/home/alper/Desktop/Works/Tez/my_mmdet3d/mmdetection3d/data/kitti/testing/velodyne_reduced/000000.bin')
inferencer = LidarDet3DInferencer('/home/alper/Desktop/Works/Tez/my_mmdet3d/mmdetection3d/work_dirs/centerpoint_005voxel_second_secfpn_4x8_cyclic_80e_kitti/centerpoint_005voxel_second_secfpn_4x8_cyclic_80e_kitti.py',
                                  '/home/alper/Desktop/Works/Tez/my_mmdet3d/mmdetection3d/work_dirs/centerpoint_005voxel_second_secfpn_4x8_cyclic_80e_kitti/epoch_40.pth')

inputs = dict(points='/home/alper/Desktop/Works/Tez/my_mmdet3d/mmdetection3d/data/kitti/training/velodyne/000009.bin')
bboxes = inferencer(inputs)


bbox3d = []
for i in range(len(bboxes["predictions"][0]["labels_3d"])):
    if bboxes["predictions"][0]["scores_3d"][i]>0.5:
        bbox3d.append(bboxes["predictions"][0]["bboxes_3d"][i])    


points = np.fromfile('/home/alper/Desktop/Works/Tez/my_mmdet3d/mmdetection3d/data/kitti/training/velodyne/000009.bin', dtype=np.float32)
points = points.reshape(-1, 4)
visualizer = Det3DLocalVisualizer()
# set point cloud in visualizer
visualizer.set_points(points)
bboxes_3d = LiDARInstance3DBoxes(torch.tensor(bbox3d))
print(bboxes_3d)
bbox_color = [[0,255,0] for i in range(bboxes_3d.shape[0])]

visualizer.draw_bboxes_3d(bboxes_3d, bbox_color)#, color="green")
"""
dataset = KittiDataset(data_root="./data/kitti", ann_file='kitti_infos_train.pkl', box_type_3d="LiDAR")#, pcd_limit_range= [0, -40, -3, 70.4, 40, 1])
infos = dataset.get_ann_info(9)

bboxes_3d2 = infos["gt_bboxes_3d"]
"""
info_file = load('/home/alper/Desktop/Works/Tez/my_mmdet3d/mmdetection3d/data/kitti/kitti_infos_train.pkl')
ann = LoadAnnotations3D(info_file)

res = list(filter(lambda person: person['sample_idx'] == 9, ann.with_bbox_3d["data_list"]))[0]

gt_bboxes =[instance["bbox_3d"] for instance in res["instances"]]

new_gt =[]
for gt in gt_bboxes:
    new_gt.append([gt[2], -gt[0], -gt[1], gt[3], gt[5], gt[4], gt[6]])

bboxes_3d2 = LiDARInstance3DBoxes(
    torch.tensor(new_gt),with_yaw=True,origin=(0.5,0.5,0))

bbox_color2 = [[255,0,0] for i in range(bboxes_3d2.shape[0])]
visualizer.draw_bboxes_3d(bboxes_3d2, bbox_color2, rot_axis=0)#, color="red")
visualizer.show()

# dataset.get_data_info(0)
# dataset.parse_data_info()
# dataset.parse_ann_info()


from mmdet3d.datasets import convert_utils

convert_utils.convert_annos(infos,0)