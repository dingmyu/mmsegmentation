export PATH=/mnt/lustre/share/gcc/gcc-5.4/bin/:$PATH
export PATH=/mnt/lustre/share/cmake-3.11.0-Linux-x86_64/bin:$PATH
export LD_LIBRARY_PATH=/mnt/lustre/share/cuda-10.1/lib64:$LD_LIBRARY_PATH
export PATH=/mnt/lustre/share/cuda-10.1/bin:$PATH
export CUDA_HOME=/mnt/lustre/share/cuda-10.1/:CUDA_HOME

conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install pytorch=1.4.0 torchvision cudatoolkit=10.1 -c pytorch

pip install mmcv-full==latest+torch1.4.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
cd mmsegmentation
pip3 install -e .


hdfs dfs -copyFromLocal ./foru hdfs:///home/byte_uslab_cvg/user/dingmingyu/dataset/
ps aux | grep hr18 | awk '{print $2}' | xargs kill -9
ps aux | grep hrnet | awk '{print $2}' | xargs kill -9
jupyter-lab --no-browser --ip=0.0.0.0 --port=9999

# US
scp -P 9001 -r tiger@10.188.180.20:/opt/tiger/uslabcv/dingmingyu/dataset/ .

# China scp -P 9001 -r tiger@10.148.57.22:/opt/tiger/uslabcv/dingmingyu/dataset/foru/ADEChallengeData2016.zip .
scp -P 9001 -r tiger@10.130.18.85:/opt/tiger/uslabcv/dingmingyu/dataset/ .
cd foru
unzip gtFine_trainvaltest.zip
unzip leftImg8bit_trainvaltest.zip
unzip ADEChallengeData2016.zip

git clone https://github.com/dingmyu/mmsegmentation.git
cd mmsegmentation
pip install -e .
mkdir data
cd data
ln -s /opt/tiger/uslabcv/dingmingyu/dataset cityscapes
cd ..
python3 tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8

python3 resize_data.py  # for small resolution

mv foru/hrnetv2_w18-00eb2006.pth ~/.cache/torch/checkpoint/
mkdir ade
ln -s /opt/tiger/uslabcv/dingmingyu/foru/ADEChallengeData2016 .