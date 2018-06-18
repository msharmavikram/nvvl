export NVVL_DIR=/nvme2/zqu/nvvl
export NVC="NVIDIA_DRIVER_CAPABILITIES=video,compute,utility"





#Experiment 1 - with resolution 540p

ROOT=/nvme2/zqu/videodata/outdata/540p/scenes/train/
for fn in 2 4 8 16 32 64
do
    echo "Running $fn with 540p resolution"
    nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader NVVL --fp16 --is_cropped --crop_size 224 224 --batchsize 8 --frames $fn
done


ROOT=/nvme2/zqu/videodata/outdata/720p/scenes/train/
for fn in 2 4 8 16 32 64
do
    echo "Running $fn with 720p resolution"
    nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader NVVL --fp16 --is_cropped --crop_size 224 224 --batchsize 8 --frames $fn
done


#This will be super time consuming. We will do it when the machine is idle.
#ROOT=/nvme2/zqu/videodata/outdata/4K/scenes/train/
#for fn in 2 4 8 16 32 64
#do
#    echo "Running $fn with 4K resolution"
#    nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader NVVL --fp16 --is_cropped --crop_size 224 224 --batchsize 8 --frames $fn
#done


#This needs to be created. use github fork - https://github.com/msharmavikram/nvvl/

ROOT=/nvme2/zqu/videodata/outdata/360p/scenes/train/
for fn in 2 4 8 16 32 64
do
    echo "Running $fn with 360p resolution"
    nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader NVVL --fp16 --is_cropped --crop_size 224 224 --batchsize 8 --frames $fn
done


ROOT=/nvme2/zqu/videodata/outdata/240p/scenes/train/
for fn in 2 4 8 16 32 64
do
    echo "Running $fn with 240p resolution"
    nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader NVVL --fp16 --is_cropped --crop_size 224 224 --batchsize 8 --frames $fn
done




#####################################################
##old
#####################################################
### FP16
#
### NVVL
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/scenes/train/
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader NVVL --fp16 --is_cropped --crop_size 224 224 --batchsize 2 --frames 4
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.05 --loader NVVL --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/scenes/train/
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader NVVL --is_cropped --crop_size 224 224 --batchsize 4 --frames 4 --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.075 --loader NVVL --is_cropped --crop_size 540 960 --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/scenes/train/
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader NVVL --is_cropped --crop_size 224 224 --batchsize 8 --frames 4  --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.13 --loader NVVL --is_cropped --crop_size 540 960 --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/scenes/train/
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader NVVL --is_cropped --crop_size 224 224 --batchsize 16 --frames 4 --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.45 --loader NVVL --is_cropped --crop_size 540 960 --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/scenes/train/
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader NVVL --is_cropped --crop_size 224 224 --batchsize 32 --frames 4 --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.45 --loader NVVL --is_cropped --crop_size 540 960 --fp16
#
### lintel
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/scenes/train/
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader lintel --fp16 --is_cropped --crop_size 224 224 --batchsize 2 --frames 4
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.27 --loader lintel --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/scenes/train/
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader lintel --is_cropped --crop_size 224 224 --batchsize 4 --frames 4 --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.27 --loader lintel --is_cropped --crop_size 540 960 --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/scenes/train/
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader lintel --is_cropped --crop_size 224 224 --batchsize 8 --frames 4 --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.28 --loader lintel --is_cropped --crop_size 540 960 --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/scenes/train/
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader lintel --is_cropped --crop_size 224 224 --batchsize 16 --frames 4 --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.45 --loader lintel --is_cropped --crop_size 540 960 --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/scenes/train/
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader lintel --is_cropped --crop_size 224 224 --batchsize 32 --frames 4 --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.45 --loader lintel --is_cropped --crop_size 540 960 --fp16
#
### pytorch - png
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/frames/png/train
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --fp16 --is_cropped --crop_size 224 224 --batchsize 2 --frames 4
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.27 --loader pytorch --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/frames/png/train
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --is_cropped --crop_size 224 224 --batchsize 4 --frames 4  --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.27 --loader pytorch --is_cropped --crop_size 540 960 --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/frames/png/train
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --is_cropped --crop_size 224 224 --batchsize 16 --frames 4 --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.28 --loader pytorch --is_cropped --crop_size 540 960 --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/frames/png/train
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --is_cropped --crop_size 224 224 --batchsize 32 --frames 4 --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.45 --loader pytorch --is_cropped --crop_size 540 960 --fp16
#
### pytorch - jpg
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/frames/jpg/train
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --fp16 --is_cropped --crop_size 224 224 --batchsize 2 --frames 4
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.27 --loader pytorch --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/frames/jpg/train
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --is_cropped --crop_size 224 224 --batchsize 4 --frames 4 --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.27 --loader pytorch --is_cropped --crop_size 540 960 --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/frames/jpg/train
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --is_cropped --crop_size 224 224 --batchsize 16 --frames 4 --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.28 --loader pytorch --is_cropped --crop_size 540 960 --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/frames/jpg/train
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --is_cropped --crop_size 224 224 --batchsize 32 --frames 4 --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.45 --loader pytorch --is_cropped --crop_size 540 960 --fp16
#
#
### NVVL
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/scenes/train/
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader NVVL --fp16 --is_cropped --crop_size 224 224 --batchsize 8 --frames 2
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.05 --loader NVVL --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/scenes/train/
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader NVVL --is_cropped --crop_size 224 224 --batchsize 8 --frames 16 --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.075 --loader NVVL --is_cropped --crop_size 540 960 --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/scenes/train/
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader NVVL --is_cropped --crop_size 224 224 --batchsize 8 --frames 32  --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.13 --loader NVVL --is_cropped --crop_size 540 960 --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/scenes/train/
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader NVVL --is_cropped --crop_size 224 224 --batchsize 8 --frames 8 --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.45 --loader NVVL --is_cropped --crop_size 540 960 --fp16
#
### lintel
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/scenes/train/
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader lintel --fp16 --is_cropped --crop_size 224 224 --batchsize 8 --frames 2
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.27 --loader lintel --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/scenes/train/
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader lintel --is_cropped --crop_size 224 224 --batchsize 8 --frames 16 --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.27 --loader lintel --is_cropped --crop_size 540 960 --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/scenes/train/
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader lintel --is_cropped --crop_size 224 224 --batchsize 8 --frames 32 --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.28 --loader lintel --is_cropped --crop_size 540 960 --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/scenes/train/
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader lintel --is_cropped --crop_size 224 224 --batchsize 8 --frames 8 --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.45 --loader lintel --is_cropped --crop_size 540 960 --fp16
#
### pytorch - png
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/frames/png/train
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --fp16 --is_cropped --crop_size 224 224 --batchsize 8 --frames 2
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.27 --loader pytorch --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/frames/png/train
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --is_cropped --crop_size 224 224 --batchsize 8 --frames 16  --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.27 --loader pytorch --is_cropped --crop_size 540 960 --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/frames/png/train
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --is_cropped --crop_size 224 224 --batchsize 8 --frames 32 --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.28 --loader pytorch --is_cropped --crop_size 540 960 --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/frames/png/train
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --is_cropped --crop_size 224 224 --batchsize 8 --frames 8 --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.45 --loader pytorch --is_cropped --crop_size 540 960 --fp16
#
### pytorch - jpg
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/frames/jpg/train
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --fp16 --is_cropped --crop_size 224 224 --batchsize 8 --frames 2
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.27 --loader pytorch --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/frames/jpg/train
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --is_cropped --crop_size 224 224 --batchsize 8 --frames 8 --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.27 --loader pytorch --is_cropped --crop_size 540 960 --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/frames/jpg/train
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --is_cropped --crop_size 224 224 --batchsize 8 --frames 16 --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.28 --loader pytorch --is_cropped --crop_size 540 960 --fp16
#
#ROOT=/nvme2/zqu/videodata/outdata/1080p/frames/jpg/train
#nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --is_cropped --crop_size 224 224 --batchsize 8 --frames 32 --fp16
##nvidia-docker run --rm  --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.45 --loader pytorch --is_cropped --crop_size 540 960 --fp16
