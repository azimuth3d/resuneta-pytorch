## ResUnet-A pytorch implmentation


## Tensorboard
    $ tensorboard --logdir [log directory]/[scope name]/[data name] \
                  --port [(optional) 4 digit port number]
---
    $ tensorboard --logdir ./log/unet/em \
                  --port 6006
                  
After the above comment executes, go **http://localhost:6006**

* You can change **[(optional) 4 digit port number]**.
* Default 4 digit port number is **6006**.


## Training with example dataset

```
python main.py --mode train \
                     --scope ResUnetA \
                     --name_data poc \
                     --dir_data ./datasets-rgb \
                     --dir_log ./log \
                     --batch_size 1 \
                     --dir_checkpoint ./checkpoints \
                     --gpu_ids 0  \
                     --num_epoch 20 \
                     --dir_result ./results \
                     --nx_in 224 \
                     --ny_in 224 \
                     --ny_out 224 \
                     --nch_in 1 \
                     --nch_out 1 \
                     --nx_out 224 \
                     --nx_load 224 \
                     --ny_load 224          
```

## Directories structure
    +---[dir_checkpoint]
    |   \---[scope]
    |       \---[name_data]
    |           +---model_epoch00000.pth
    |           |   ...
    |           \---model_epoch12345.pth
    +---[dir_data]
    |   \---[name_data]
    |       +---test
    |       |   +---00000.npy
    |       |   |   ...
    |       |   \---12345.npy
    |       +---train
    |       |   +---00000.npy
    |       |   |   ...
    |       |   \---12345.npy
    |       \---val
    |           +---00000.npy
    |           |   ...
    |           \---12345.npy
    +---[dir_log]
    |   \---[scope]
    |       \---[name_data]
    |           +---arg.txt
    |           \---events.out.tfevents
    \---[dir_result]
        \---[scope]
            \---[name_data]
                +---images
                |   +---00000-input.png
                |   +---00000-label.png
                |   +---00000-output.png
                |   |   ...
                |   +---12345-input.png
                |   +---12345-label.png

* Above directory is created by setting arguments when **main.py** is executed.               
