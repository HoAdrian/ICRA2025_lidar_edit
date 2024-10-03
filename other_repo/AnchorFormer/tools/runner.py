import numpy as np
import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import open3d
from datasets.data_transforms import NormalizeObjectPose

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # Build Dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = \
        builder.dataset_builder(args, config.dataset.train), builder.dataset_builder(args, config.dataset.val)
    # Build Model
    base_model = builder.model_builder(config.model)

    if args.use_gpu:
        base_model.to(args.local_rank)
        
    # Parameter Setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # Resume Ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, \
                                                         device_ids=[args.local_rank % torch.cuda.device_count()], \
                                                         find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
        
    # Optimizer & Scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # Training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['SparseLoss', 'DenseLoss', 'SparsePenalty', 'DensePenalty'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
                if config.dataset.train._base_.CARS:
                    if idx == 0:
                        print_log('padding while KITTI training', logger=logger)
                    partial = misc.random_dropping(partial, epoch) # specially for KITTI finetune

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial = partial.cuda()
                
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            num_iter += 1
           
            ret = base_model(partial)
            
            sparse_loss, dense_loss = base_model.module.get_loss(ret, gt)
            sparse_penalty, dense_penalty = base_model.module.get_penalty()
            sparse_loss = config.loss.sparse_loss_weight * sparse_loss
            dense_loss = config.loss.dense_loss_weight * dense_loss
            
            sparse_penalty = config.loss.sparse_penalty_weight * sparse_penalty
            dense_penalty = config.loss.dense_penalty_weight * dense_penalty
            _loss = sparse_loss + dense_loss + sparse_penalty + dense_penalty
            _loss.backward()

            # Forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                dense_loss = dist_utils.reduce_tensor(dense_loss, args)
                sparse_penalty = dist_utils.reduce_tensor(sparse_penalty, args)
                dense_penalty = dist_utils.reduce_tensor(dense_penalty, args)
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000, \
                               sparse_penalty.item() * 1000, dense_penalty.item() * 1000])
            else:
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000, \
                               sparse_penalty.item() * 1000, dense_penalty.item() * 1000])


            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)
                train_writer.add_scalar('LR/training', optimizer.param_groups[0]['lr'], n_itr)
                train_writer.add_scalar('Penalty/Batch/Sparse', sparse_penalty.item() * 1000, n_itr)
                train_writer.add_scalar('Penalty/Batch/Dense', dense_penalty.item() * 1000, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch)
            train_writer.add_scalar('Penalty/Epoch/Sparse', losses.avg(2), epoch)
            train_writer.add_scalar('Penalty/Epoch/Dense', losses.avg(3), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger)

            # Save checkpoints
            if  metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        if (config.max_epoch - epoch) < 10:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger) 
            
    train_writer.close()
    val_writer.close()

    
def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial = partial.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            ret = base_model(partial)
            coarse_points = ret[0]
            dense_points = ret[1]

            sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
            sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
            dense_loss_l1 =  ChamferDisL1(dense_points, gt)
            dense_loss_l2 =  ChamferDisL2(dense_points, gt)

            if args.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, \
                                dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

            _metrics = Metrics.get(dense_points, gt) 

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            if val_writer is not None and idx % args.val_interval == 0:
                input_pc = partial.squeeze().detach().cpu().numpy()
                input_pc = misc.get_ptcloud_img(input_pc)
                val_writer.add_image('Model%02d-%d/Input'% (idx, epoch) , input_pc, epoch, dataformats='HWC')

                sparse = coarse_points.squeeze().cpu().numpy()
                sparse_img = misc.get_ptcloud_img(sparse)
                val_writer.add_image('Model%02d-%d/Sparse' % (idx, epoch), sparse_img, epoch, dataformats='HWC')
                pred_sparse_img = misc.get_ordered_ptcloud_img(sparse[0:224,:])
                val_writer.add_image('Model%02d-%d/PredSparse' % (idx, epoch), pred_sparse_img, epoch, dataformats='HWC')

                dense = dense_points.squeeze().cpu().numpy()
                dense_img = misc.get_ptcloud_img(dense)
                val_writer.add_image('Model%02d-%d/Dense' % (idx, epoch), dense_img, epoch, dataformats='HWC')
                
                gt_ptcloud = gt.squeeze().cpu().numpy()
                gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud)
                val_writer.add_image('Model%02d-%d/DenseGT' % (idx, epoch), gt_ptcloud_img, epoch, dataformats='HWC')
        
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
                
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
     
    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch)
        val_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(2), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())


crop_ratio = {
    'easy': 1/4,
    'median' :1/2,
    'hard':3/4
}

def test_net(args, config, test_writer=None):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
 
    base_model = builder.model_builder(config.model)
    print_log(base_model, logger = logger)
    
    # load checkpoints
    # builder.load_model(base_model, args.ckpts, logger = logger)
    state_dict = torch.load(args.ckpts, map_location='cpu')
    base_model.load_state_dict(state_dict)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, test_writer, args, config, logger=logger)

def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, test_writer, args, config, logger = None):

    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1
    
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data, old_data) in enumerate(test_dataloader): # I added old data for unnormalizing the output point cloud of kitti
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", taxonomy_ids)
            if isinstance(taxonomy_ids[0], tuple):
                category_foldername, sample_num = taxonomy_ids
                category_foldername = category_foldername[0]
                sample_num = sample_num[0]
            else:
                taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            
            model_id = model_ids[0]
            #print("....... unstransformed data", old_data)

            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
                
                
                ret = base_model(partial)
                coarse_points = ret[0]
                dense_points = ret[1]
                                         
                sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, \
                                    dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                _metrics = Metrics.get(dense_points ,gt)
                test_metrics.update(_metrics)

                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                            torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]
                num_crop = int(npoints * crop_ratio[args.mode])
                for item in choice:           
                    partial, _ = misc.seprate_point_cloud(gt, npoints, num_crop, fixed_points = item)
                    # NOTE: subsample the input
                    partial = misc.fps(partial, 2048)
                    ret = base_model(partial)
                    coarse_points = ret[0]
                    dense_points = ret[1]

                    sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                    sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                    dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                    dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                    test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, \
                                        dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                    _metrics = Metrics.get(dense_points ,gt)

                    # test_metrics.update(_metrics)

                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                    category_metrics[taxonomy_id].update(_metrics)
                    
            elif dataset_name == 'KITTI' or dataset_name=='Custom':
                partial = data.cuda()
                ret = base_model(partial)
                dense_points = ret[1]
                target_path = os.path.join(args.experiment_path, 'vis_result')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                # misc.visualize_KITTI(
                #     os.path.join(target_path, f'{model_id}_{idx:03d}'),
                #     [partial[0].cpu(), dense_points[0].cpu()]
                # )

                if dataset_name=='KITTI':
                    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ VISUALIZE ..... ")
                    dense_dir = f"/home/shinghei/lidar_generation/AnchorFormer/test_results_KITTI/{taxonomy_id}/dense"
                    sparse_dir = f"/home/shinghei/lidar_generation/AnchorFormer/test_results_KITTI/{taxonomy_id}/sparse"
                    os.makedirs(dense_dir, exist_ok=True)
                    os.makedirs(sparse_dir, exist_ok=True)

                    ### convert the transformed point cloud back to the original coordinate system
                    original_bbox = old_data["bounding_box"][0].cpu().numpy()
                    print("... BOX: ", original_bbox.shape)
                    unnormalized_partial = NormalizeObjectPose.inverse(ptcloud=partial[0].cpu().numpy(), bbox=original_bbox)
                    unnormalized_dense = NormalizeObjectPose.inverse(ptcloud=dense_points[0].cpu().numpy(), bbox=original_bbox)

                    

                    # save input pcd
                    pcd_sparse = open3d.geometry.PointCloud()
                    pcd_sparse.points = open3d.utility.Vector3dVector(unnormalized_partial)
                    #pcd_sparse.points = open3d.utility.Vector3dVector(partial[0].cpu().numpy())

                    pcd_dense = open3d.geometry.PointCloud()
                    pcd_dense.points = open3d.utility.Vector3dVector(unnormalized_dense)

                    # open3d.visualization.draw_geometries([pcd_sparse, pcd_dense.translate((1,0,0))]) 
                    corners = original_bbox

                    pcd0 = open3d.geometry.PointCloud()
                    pcd0.points = open3d.utility.Vector3dVector(corners[0:1, :])
                    pcd_colors = np.tile(np.array([[0,0,0]]), (1, 1))
                    pcd0.colors = open3d.utility.Vector3dVector(pcd_colors)

                    pcd1 = open3d.geometry.PointCloud()
                    pcd1.points = open3d.utility.Vector3dVector(corners[1:2, :])
                    pcd_colors = np.tile(np.array([[0,1,0]]), (1, 1))
                    pcd1.colors = open3d.utility.Vector3dVector(pcd_colors)

                    pcd2 = open3d.geometry.PointCloud()
                    pcd2.points = open3d.utility.Vector3dVector(corners[2:3, :])
                    pcd_colors = np.tile(np.array([[1,0,0]]), (1, 1))
                    pcd2.colors = open3d.utility.Vector3dVector(pcd_colors)

                    pcd3 = open3d.geometry.PointCloud()
                    pcd3.points = open3d.utility.Vector3dVector(corners[3:4, :])
                    pcd_colors = np.tile(np.array([[1,0,0]]), (1, 1))
                    pcd3.colors = open3d.utility.Vector3dVector(pcd_colors)

                    pcd4 = open3d.geometry.PointCloud()
                    pcd4.points = open3d.utility.Vector3dVector(corners[4:5, :])
                    pcd_colors = np.tile(np.array([[0,0,1]]), (1, 1))
                    pcd4.colors = open3d.utility.Vector3dVector(pcd_colors)

                    pcd5 = open3d.geometry.PointCloud()
                    pcd5.points = open3d.utility.Vector3dVector(corners[5:6, :])
                    pcd_colors = np.tile(np.array([[0,0,1]]), (1, 1))
                    pcd5.colors = open3d.utility.Vector3dVector(pcd_colors)

                    pcd6 = open3d.geometry.PointCloud()
                    pcd6.points = open3d.utility.Vector3dVector(corners[6:7, :])
                    pcd_colors = np.tile(np.array([[0,0,1]]), (1, 1))
                    pcd6.colors = open3d.utility.Vector3dVector(pcd_colors)

                    pcd7 = open3d.geometry.PointCloud()
                    pcd7.points = open3d.utility.Vector3dVector(corners[7:8, :])
                    pcd_colors = np.tile(np.array([[0,0,1]]), (1, 1))
                    pcd7.colors = open3d.utility.Vector3dVector(pcd_colors)


                    frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))

                    open3d.visualization.draw_geometries([pcd_sparse, pcd0, pcd1, pcd2, pcd3, pcd4, pcd5, pcd6, pcd7, frame]) 

                    open3d.io.write_point_cloud(f"{sparse_dir}/sample_{idx}.pcd", pcd_sparse)
                    open3d.io.write_point_cloud(f"{dense_dir}/sample_{idx}.pcd",pcd_dense)

                else:
                    # category_foldername, sample_num = taxonomy_ids
                    # category_foldername = category_foldername[0]
                    # sample_num = sample_num[0]

                    print("HHHHHHHHIIIIIIIIIIIIIIIIIIIIIIi")

                    save_root = f"/home/shinghei/lidar_generation/our_ws/foreground_object_pointclouds/dense_nusc/{category_foldername}"
                    os.makedirs(save_root, exist_ok=True)

                    ### convert the transformed point cloud back to the original coordinate system
                    original_bbox = old_data["bounding_box"][0].cpu().numpy()
                    #print("... BOX: ", original_bbox.shape)
                    normalized_partial = partial[0].cpu().numpy()
                    normalized_dense = dense_points[0].cpu().numpy()
                    unnormalized_partial = NormalizeObjectPose.inverse(ptcloud=partial[0].cpu().numpy(), bbox=original_bbox)
                    unnormalized_dense = NormalizeObjectPose.inverse(ptcloud=dense_points[0].cpu().numpy(), bbox=original_bbox)

                    if category_foldername=="car" or True:
                        # save input pcd
                        pcd_sparse = open3d.geometry.PointCloud()
                        pcd_sparse.points = open3d.utility.Vector3dVector(unnormalized_partial)
                        pcd_sparse.colors = open3d.utility.Vector3dVector(np.tile(np.array([[0,0,1]]), (len(unnormalized_partial), 1)))

                        pcd_dense = open3d.geometry.PointCloud()
                        pcd_dense.points = open3d.utility.Vector3dVector(unnormalized_dense)

                        # open3d.visualization.draw_geometries([pcd_dense, pcd_sparse.translate((1,0,0))]) 
                        #open3d.visualization.draw_geometries([pcd_dense, pcd_sparse.translate((8,0,0))]) 

                        open3d.io.write_point_cloud(f"{save_root}/sample_{sample_num}.pcd",pcd_dense)

                    




                
                continue
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')
            
            # Visualize
            if test_writer is not None and idx % args.test_interval == 0:                
                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ VISUALIZE ..... ")
                # input_pc = partial.squeeze().detach().cpu().numpy()
                # input_img = misc.get_ptcloud_img(input_pc)
                # test_writer.add_image('Model%02d-test/Input'% idx , input_img, dataformats='HWC')

                # # save input pcd
                # pcd_sparse = open3d.geometry.PointCloud()
                # pcd_sparse.points = open3d.utility.Vector3dVector(np.array(input_pc))
                # open3d.io.write_point_cloud(f"/home/shinghei/lidar_generation/AnchorFormer/test_results/input_{taxonomy_id}.pcd", pcd_sparse)

                # sparse = coarse_points.squeeze().cpu().numpy()
                # sparse_img = misc.get_ptcloud_img(sparse)
                # test_writer.add_image('Model%02d-test/Sparse' % idx, sparse_img, dataformats='HWC')

                # dense = dense_points.squeeze().cpu().numpy()
                # dense_img = misc.get_ptcloud_img(dense)
                # test_writer.add_image('Model%02d-test/Dense' % idx, dense_img, dataformats='HWC')
                
                # # save dense pcd
                # pcd_sparse = open3d.geometry.PointCloud()
                # pcd_sparse.points = open3d.utility.Vector3dVector(np.array(dense))
                # open3d.io.write_point_cloud(f"/home/shinghei/lidar_generation/AnchorFormer/test_results/dense_{taxonomy_id}.pcd",pcd_sparse)
                
                # gt_ptcloud = gt.squeeze().cpu().numpy()
                # gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud)
                # test_writer.add_image('Model%02d-test/DenseGT' % idx, gt_ptcloud_img, dataformats='HWC')
                    
                # Save output results
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
                
        # Compute testing results
        # if dataset_name == 'KITTI':
        #     return
        # for _,v in category_metrics.items():
        #     test_metrics.update(v.avg())
        # print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)

     

    # # Print testing results
    # shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    # print_log('============================ TEST RESULTS ============================',logger=logger)
    # msg = ''
    # msg += 'Taxonomy\t'
    # msg += '#Sample\t'
    # for metric in test_metrics.items:
    #     msg += metric + '\t'
    # msg += '#ModelName\t'
    # print_log(msg, logger=logger)


    # for taxonomy_id in category_metrics:
    #     msg = ''
    #     msg += (taxonomy_id + '\t')
    #     msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
    #     for value in category_metrics[taxonomy_id].avg():
    #         msg += '%.3f \t' % value
    #     msg += shapenet_dict[taxonomy_id] + '\t'
    #     print_log(msg, logger=logger)

    # msg = ''
    # msg += 'Overall \t\t'
    # for value in test_metrics.avg():
    #     msg += '%.3f \t' % value
    # print_log(msg, logger=logger)
    return 


def test_nuscenes(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, test_writer, args, config, logger = None):

    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1
    
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            category_foldername, sample_num = taxonomy_ids
            category_foldername = category_foldername[0]
            sample_num = sample_num[0]
            model_id = model_ids[0]

            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'PCN' or 'Custom':
                partial = data[0].cuda()
                gt = data[1].cuda()
                
                
                ret = base_model(partial)
                coarse_points = ret[0]
                dense_points = ret[1]
                                         
                sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, \
                                    dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                _metrics = Metrics.get(dense_points ,gt)
                test_metrics.update(_metrics)

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                            torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]
                num_crop = int(npoints * crop_ratio[args.mode])
                for item in choice:           
                    partial, _ = misc.seprate_point_cloud(gt, npoints, num_crop, fixed_points = item)
                    # NOTE: subsample the input
                    partial = misc.fps(partial, 2048)
                    ret = base_model(partial)
                    coarse_points = ret[0]
                    dense_points = ret[1]

                    sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                    sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                    dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                    dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                    test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, \
                                        dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                    _metrics = Metrics.get(dense_points ,gt)

                    # test_metrics.update(_metrics)
                    
            elif dataset_name == 'KITTI':
                partial = data.cuda()
                ret = base_model(partial)
                dense_points = ret[1]
                target_path = os.path.join(args.experiment_path, 'vis_result')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                misc.visualize_KITTI(
                    os.path.join(target_path, f'{model_id}_{idx:03d}'),
                    [partial[0].cpu(), dense_points[0].cpu()]
                )
                continue
            else:
                print("GOOD")
                #raise NotImplementedError(f'Train phase do not support {dataset_name}')
            
            # Visualize
            if test_writer is not None and idx % args.test_interval == 0:                
                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ VISUALIZE ..... ")
                
                save_root = f"/home/shinghei/lidar_generation/our_ws/foreground_object_pointclouds/dense_nusc/{category_foldername}"
                os.makedirs(save_root, exist_ok=True)

                sparse = coarse_points.squeeze().cpu().numpy()
                dense = dense_points.squeeze().cpu().numpy()
                
                # save dense pcd
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(np.array(dense))
                # if category_foldername=="car":
                #     open3d.visualization.draw_geometries([pcd]) 
                open3d.io.write_point_cloud(f"{save_root}/sample_{sample_num}.pcd",pcd)
                
                