from detrex.config import get_config
from dino_r50_MHS_mixFFN import model


dataloader = get_config("common/data/pvel.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train

train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/mix-dino"

train.max_iter = 60000
train.eval_period = 5000
train.log_period = 50
train.checkpointer.period = 5000

train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

train.device = "cuda"
model.device = train.device

optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

dataloader.train.num_workers = 4
dataloader.train.total_batch_size = 4
dataloader.evaluator.output_dir = train.output_dir