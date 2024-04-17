import mmdet3d
import numpy as np
from mmdet3d.datasets import KittiDataset


kitti_root = './data/kitti/training'

data = KittiDataset(kitti_root, ann_file="kitti_infos_train.pkl")
sample = data.get_sample(0)



# Model yapılandırması ve ağırlıkları yükleyin
config_file = './work_dirs/centerpoint_005voxel_second_secfpn_4x8_cyclic_80e_kitti/centerpoint_005voxel_second_secfpn_4x8_cyclic_80e_kitti.py'
checkpoint_file = './work_dirs/centerpoint_005voxel_second_secfpn_4x8_cyclic_80e_kitti/epoch_40.pth'
model = init_model(config_file, checkpoint_file, device='cuda:0')


# İstediğiniz örnek ID'leri
sample_ids = [50]  # Örnek olarak 3 ID

# KITTI dataset yollarınız

for sample_id in sample_ids:
    # Veri dosya yolu
    img_file = f'{kitti_root}/image/{sample_id:06d}.png'
    lidar_file = f'{kitti_root}/velodyne/{sample_id:06d}.bin'
    calib_file = f'{kitti_root}/calib/{sample_id:06d}.txt'
    
    # Çıkarım yapın
    result, data = inference_detector(model, lidar_file)
    visualizer = Det3DLocalVisualizer()
    visualizer.add_datasample('3D Scene', data_input,gt_det3d_data_sample,vis_task='lidar_seg')
    visualizer.show()

    # Gerçek bbox'lar ve tahminler ile görselleştirme
