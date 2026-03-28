from cleanfid import fid
import torch
from pytorch_fid import fid_score

fdir1 = "/home/sadong/refine_ddpm/paint/spadecross1_regionloss1.0,1.2,1.4,1.4,1.4_2900_maskvae/result"
#fdir1="/home/sadong/refine_ddpm/results/focal_padecrossnew1/condition"
#fdir1="/home/sadong/refine_ddpm/results/focal_spadecross1_regionloss1.0,1.2,1.4,1.4,1.4_2720/condition"
#fdir1="/home/sadong/refine_ddpm/paint/spadecross1_regionloss1.0,1.2,1.4,1.4,1.4_2900_refine-spadesam_2000_removesmall80_detectormask/result"
fdir2="/home/sadong/refine_ddpm/test_orig"
# score = fid.compute_kid(fdir1, fdir2,device=torch.device("cuda:1"),use_dataparallel=False,z_dim=2048,batch_size=8)
# print(score)
fid_value = fid_score.calculate_fid_given_paths(paths=[fdir2, fdir1],
                                                batch_size=8,
                                                device=torch.device("cuda:1"),
                                                dims=2048
                                                )
print('FID value:', fid_value)
