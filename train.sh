CUDA_VISIBLE_DEVICES=2,3 PORT=9000 ./tools/dist_train.sh  configs/hrnet/mingyu_fcn_hr18_512x512_80k_ade20k.py 2 --work_dir ./output/mingyu_fcn_hr18_512x512_160k_ade20k
#CUDA_VISIBLE_DEVICES=0 PORT=9002 ./tools/dist_train.sh  configs/hrnet/mingyu_fcn_hrnet_512x512_80k_ade20k.py 1 --work_dir ./output/mingyu_fcn_hrnet_512x512_80k_ade20k
#CUDA_VISIBLE_DEVICES=1 PORT=9003 ./tools/dist_train.sh  configs/hrnet/mingyu_fcn_hrnet_sync_512x512_80k_ade20k.py 1 --work_dir ./output/mingyu_fcn_hrnet_sync_512x512_80k_ade20k
#CUDA_VISIBLE_DEVICES=2,3 PORT=9000 ./tools/dist_train.sh  configs/hrnet/mingyu_fcn_hr18_512x1024_40k_cityscapes.py 2 --work_dir ./output/mingyu_fcn_hrnet_sync_512x1024_40k_cityscapes_pretrained
