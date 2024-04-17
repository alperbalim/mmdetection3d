import torch
import numpy as np
from mmdet3d.apis import LidarDet3DInferencer
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes
from mmengine import load

#dict(points='/home/alper/Desktop/Works/Tez/my_mmdet3d/mmdetection3d/data/kitti/testing/velodyne_reduced/000000.bin')
inferencer = LidarDet3DInferencer('/home/alper/Desktop/Works/Tez/my_mmdet3d/mmdetection3d/work_dirs/centerpoint_005voxel_second_secfpn_4x8_cyclic_80e_kitti/centerpoint_005voxel_second_secfpn_4x8_cyclic_80e_kitti.py',
                                  '/home/alper/Desktop/Works/Tez/my_mmdet3d/mmdetection3d/work_dirs/centerpoint_005voxel_second_secfpn_4x8_cyclic_80e_kitti/epoch_40.pth')

inputs = dict(points='/home/alper/Desktop/Works/Tez/my_mmdet3d/mmdetection3d/data/kitti/training/velodyne/000009.bin')
bboxes = inferencer(inputs)

bbox3d = []
for i in range(len(bboxes["predictions"][0]["labels_3d"])):
    if bboxes["predictions"][0]["scores_3d"][i]>0.7:
        bbox3d.append(bboxes["predictions"][0]["bboxes_3d"][i])    

points = np.fromfile('/home/alper/Desktop/Works/Tez/my_mmdet3d/mmdetection3d/data/kitti/training/velodyne/000009.bin', dtype=np.float32)
points = points.reshape(-1, 4)
visualizer = Det3DLocalVisualizer()
# set point cloud in visualizer
visualizer.set_points(points)
bboxes_3d = LiDARInstance3DBoxes(
    torch.tensor(bbox3d))
# Draw 3D bboxes
bbox_color = [[0,255,0] for i in range(bboxes_3d.shape[0])]
#visualizer.draw_bboxes_3d(bboxes_3d, bbox_color)#, color="green")

info_file = load('/home/alper/Desktop/Works/Tez/my_mmdet3d/mmdetection3d/data/kitti/kitti_infos_train.pkl')
res = list(filter(lambda person: person['sample_idx'] == 9, info_file["data_list"]))[0]
bboxes_3d2 = LiDARInstance3DBoxes(
    torch.tensor([instance["bbox_3d"] for instance in res["instances"]]))

bbox_color2 = [[255,0,0] for i in range(bboxes_3d2.shape[0])]
visualizer.draw_bboxes_3d(bboxes_3d2, bbox_color2)#, color="red")
visualizer.show()