import torch 
import os

weight_path = "singlesweep_checkpoint_epoch_20.pth"  #"cbgs_pp_centerpoint_nds6070.pth"
checkpoint = torch.load(weight_path)
epoch = checkpoint.get('epoch', -1)
print(f"epoch: {checkpoint.keys()}")

print(f"optimizer: ",checkpoint["optimizer_state"])


# def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
#     if not os.path.isfile(filename):
#         raise FileNotFoundError

#     logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
#     loc_type = torch.device('cpu') if to_cpu else None
#     checkpoint = torch.load(filename, map_location=loc_type)
#     epoch = checkpoint.get('epoch', -1)
#     it = checkpoint.get('it', 0.0)

#     self._load_state_dict(checkpoint['model_state'], strict=True)

#     if optimizer is not None:
#         if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
#             logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
#                         % (filename, 'CPU' if to_cpu else 'GPU'))
#             optimizer.load_state_dict(checkpoint['optimizer_state'])
#         else:
#             assert filename[-4] == '.', filename
#             src_file, ext = filename[:-4], filename[-3:]
#             optimizer_filename = '%s_optim.%s' % (src_file, ext)
#             if os.path.exists(optimizer_filename):
#                 optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
#                 optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

#     if 'version' in checkpoint:
#         print('==> Checkpoint trained from version: %s' % checkpoint['version'])
#     logger.info('==> Done')

#     return it, epoch
