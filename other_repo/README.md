# AnchorFormer
- datasets/customDataset.py, cfgs/custom_models/AnchorFormer.yaml, dataset_configs/custom.yaml, tools/runner.py are custom scripts for running point cloud completion on partial-view vehicle point cloud extracted from NuScenes. Here is the original repo: https://github.com/chenzhik/AnchorFormer and where I downloaded the dataset: https://github.com/yuxumin/PoinTr/blob/master/DATASET.md


# OpenPCDet
- We train and evaluate VoxelNext on single-sweep lidar point cloud without intensity information. We only concern three classes: car, truck and bus.

## custom codes:
- configs: custom_cbgs_voxel0075_voxelnext.yaml, custom_nuscenes_dataset.yaml
- data processing: custom_nuscenes_dataset.py, fill_trainval_infos method in nuscnes_utils.py
- In pcdet/datasets/augmentor/database_sampler.py, we set class_names = {"car", "bus", "truck"} when training with our synthetic data

## running experiments
Run run_openpcdet.sh (uncomment any useful lines to run)
There are 3 main files: 
- data processing: pcdet.datasets.nuscenes.custom_nuscenes_dataset
- tools/train: training on the processed data
- tools/test: testing on the processed data

Example:
```
python -m pcdet.datasets.nuscenes.custom_nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/custom_nuscenes_dataset.yaml --version $VERSION

python tools/train.py --cfg_file ${CONFIG_FILE}

python tools/test.py --cfg_file ${CONFIG_FILE} --batch_size 1 --ckpt ${CHECKPOINT_FILE}
```