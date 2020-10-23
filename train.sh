#CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=9000 ./tools/dist_train.sh  configs/hrnet/cityscapes_fcn_hr18_512x1024_40k.py 4 --work_dir ./output/cityscapes_hr18
#CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=9000 ./tools/dist_train.sh  configs/hrnet/ade20k_fcn_hr18_512x512_80k.py 4 --work_dir ./output/ade20k_hr18
#CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=9000 ./tools/dist_train.sh  configs/ade20k_hrnet_512x512_80k.py 4 --work_dir ./output/mingyu_ade20k
#CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=9000 ./tools/dist_train.sh  configs/cityscapes_hrnet_512x1024_40k.py 4 --work_dir ./output/mingyu_cityscapes

#PORT=9099 ./tools/dist_train.sh  configs/cityscapes_hrnet_512x1024_40k.py 8 --work_dir ./output/hrnet_cityscapes # --resume-from ./output/hrnet_cityscapes/iter_20000.pth

PORT=9099 ./tools/dist_train.sh  configs/res50_psp_512x1024_40k.py 8 --work_dir ./output/resnet_cityscapes
PORT=9099 ./tools/dist_train.sh  configs/res18_psp_512x1024_40k.py 8 --work_dir ./output/resnet_cityscapes
PORT=9099 ./tools/dist_train.sh  configs/shufflenet_psp_512x1024_40k.py 8 --work_dir ./output/shufflenet_cityscapes
