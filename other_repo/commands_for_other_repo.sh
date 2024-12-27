
###################################################################
################ Tensor Board Visualization ##########################
######################################################################

### open tensor board visualization for tensorboard events saved in a folder named train
######### option 1: Run TensorBoard with a public IP (ignore this option)
# #1) on computer X
# tensorboard --logdir=train --host 0.0.0.0 --port 6006
# #Find the IP address of computer X and make sure the port 6006 is open on computer X's firewall.
# #2) on computer Y
# #Access TensorBoard from computer Y by going to http://<IP_of_computer_X>:6006 in the web browser on computer Y, replacing <IP_of_computer_X> with computer X's actual IP address.
# http://73.191.194.161:6006

############# option 2: ssh and do port forwarding
#1) on computer shinghei@73.191.194.161
tensorboard --logdir=train
#2) on computer Y
ssh -L 6006:localhost:6006 shinghei@73.191.194.161
#Access TensorBoard from computer Y by going to http://localhost:6006/ in the web browser on computer Y, replacing <IP_of_computer_X> with computer X's actual IP address.
# check ip address tensor board is bound to run 
lsof -i :6006



###################################################################
################ PoinTr ##########################
######################################################################
#inference_vehicles.py is a custom file made for shinghei point cloud completion

#### point cloud completion inference on our foreground vehicles extracted from NuScenes
python tools/inference_vehicles.py cfgs/PCN_models/PoinTr.yaml PCN_Pretrained.pth --pc_root /home/shinghei/lidar_generation/Lidar_generation/foreground_object_pointclouds --save_vis_img --out_pc_root /home/shinghei/lidar_generation/hehe
python tools/inference_vehicles.py cfgs/KITTI_models/PoinTr.yaml KITTI.pth --pc_root /home/shinghei/lidar_generation/Lidar_generation/foreground_object_pointclouds --save_vis_img --out_pc_root /home/shinghei/lidar_generation/hehe
python tools/inference_vehicles.py cfgs/ShapeNet55_models/PoinTr.yaml pointr_training_from_scratch_c55_best.pth --pc_root /home/shinghei/lidar_generation/Lidar_generation/foreground_object_pointclouds --save_vis_img --out_pc_root /home/shinghei/lidar_generation/hehe


###################################################################
################ AnchorFormer ##########################
######################################################################

#### point cloud completion inference on our foreground vehicles extracted from NuScenes
CUDA_VISIBLE_DEVICES=0 python main.py --ckpts anchorformer_dict.pth --config ./cfgs/custom_models/AnchorFormer.yaml --exp_name test_ckpt --test

#### point cloud completion on PCN and KITTI dataset for debugging
CUDA_VISIBLE_DEVICES=0 python main.py --ckpts anchorformer_dict.pth --config ./cfgs/PCN_models/AnchorFormer.yaml --exp_name test_ckpt --test
CUDA_VISIBLE_DEVICES=0 python main.py --ckpts anchorformer_dict.pth --config ./cfgs/KITTI_models/AnchorFormer.yaml --exp_name train_kitti_ckpt --test

####Training: 
#CUDA_VISIBLE_DEVICES=0 python main.py --ckpts anchorformer_dict.pth --config ./cfgs/KITTI_models/AnchorFormer.yaml --exp_name train_kitti_ckpt --test

CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 python -m torch.distributed.launch --node_rank=0 --nnodes=1 --master_port=13232 --nproc_per_node=1 main.py --launcher pytorch --sync_bn --config ./cfgs/ShapeNet55_models/AnchorFormer.yaml --exp_name try_to_train_anchorformer --val_freq 10 --val_interval 50 
#CUDA_VISIBLE_DEVICES=0 python --config ./cfgs/ShapeNet55_models/AnchorFormer.yaml --tfboard_path shapenet_tfboard/ --exp_name try_to_train_anchorformer --val_freq 10 --val_interval 50 


########################################################################
###########Commands for OpenPCDet is in that directory #################
########################################################################