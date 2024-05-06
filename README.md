1 Follow DETREX to build the env
2 run ```python tools/train_net.py
--config-file path_to_config 
--num-gpus 1``` to train
3 run ```python tools/train_net.py 
--eval 
--config-file path_to_config 
--num-gpus 1 train.init_checkpoint="path_to_pth"``` to evlauate
4 the proposed dataset is available at https://github.com/SeveralYang/elpv-detection-dataset
