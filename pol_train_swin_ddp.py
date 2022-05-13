import cv2
import glob
import json
import numpy as np
import os
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from albumentations import (
    Compose, 
    ColorJitter, 
    Affine, 
    Perspective,
    Downscale,
    ChannelShuffle, 
    Normalize,
    Resize,
    FancyPCA
)
from albumentations.pytorch import ToTensorV2
from dataclasses import dataclass
from datetime import datetime
from PIL import (
    Image, 
    ImageFile
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)
from torch_optimizer import Lamb
from transformers import (
    SwinModel,
    SwinPreTrainedModel,
    get_cosine_schedule_with_warmup
)
from tqdm import (
    tqdm,
    trange
)


@dataclass(
    init=False, 
    repr=True, 
    eq=True, 
    order=False, 
    unsafe_hash=False, 
    frozen=True
)
class Config():
    SWIN_CHECKPOINT = "microsoft/swin-base-patch4-window12-384"
    EXP_NAME = "Swin_Political"
    SEED = 77

    DATA_SUBSET_PATH_TRAIN = "/mnt/staging/Downloads/Final/Train"
    DATA_SUBSET_PATH_VAL = "/mnt/staging/Downloads/Final/Val"
    DATA_SUBSET_SIZE_TRAIN = None
    DATA_SUBSET_SIZE_VAL = None
    DATA_SUBSET_SHUFFLE_TRAIN = True
    DATA_SUBSET_SHUFFLE_VAL = False
    DATA_SUBSET_BATCH_SIZE_TRAIN = 32
    DATA_SUBSET_BATCH_SIZE_VAL = 32
    DATALOADER_PROCESS_COUNT = 8
    DATALOADER_PREFETCH_FACTOR = 32
    DATALOADER_PIN_MEMORY = True

    DEFAULT_LR = 5e-4
    EMBEDDING_LR = 2e-5
    ENCODER_BASE_LR = 5e-4
    ENCODER_LR_MULTIPLIER = 0.95
    LAYERNORM_LR = 1e-3
    CLASSIFIER_LR = 4e-3
    WEIGHT_DECAY = 1e-2
    EPSILON = 1e-8
    CLIP_VALUE = 4.0
    WARMUP_PROP = 0.125
    MIN_WARMUP_STEPS = 5000
    EPOCHS = 20
    NUM_CLASSES = 2
    NUM_CYCLES = 2.5
    LABEL_SMOOTHING = 0.1
    LOSS_WEIGHTS = [4.0, 1.0]


class BinaryFaceDataset(torch.utils.data.Dataset):
    def __init__(self, folder=".", transform=None, sample=None):
        negative_path_array = np.array(glob.glob(f"{folder}/0_Negative/*", recursive=False))
        negative_label_array = np.zeros(negative_path_array.shape[0], dtype=np.int8)
        positive_path_array = np.array(glob.glob(f"{folder}/1_Positive/*", recursive=False))
        positive_label_array = np.ones(positive_path_array.shape[0], dtype=np.int8)
        self.total_path_array = np.concatenate([negative_path_array, positive_path_array])
        self.total_label_array = np.concatenate([negative_label_array, positive_label_array])
        shuffle_permute = np.random.permutation(self.total_path_array.shape[0])
        self.total_path_array = np.take(self.total_path_array, shuffle_permute)
        self.total_label_array = np.take(self.total_label_array, shuffle_permute)
        
        if sample is not None:
            self.total_path_array = self.total_path_array[:sample]
            self.total_label_array = self.total_label_array[:sample]
        
        self.size = self.total_path_array.shape[0]
        
        self.transform = transform
        if self.transform is None:
            self.transform = ToTensorV2(p=1.0)
        
        print(f"Binary dataset of size {self.size} initialized from path {folder}")
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if idx > self.size:
            return ValueError("Index Out of Bounds")
        with Image.open(self.total_path_array[idx]) as image:
            try:
                image = self.transform(image = np.asarray(image))['image']
                label = int(self.total_label_array[idx])
                return (image, label)
            except Exception as e:
                return (torch.zeros([3, 384, 384]), 1)


class SwinForImageClassificationFixed(SwinPreTrainedModel):
    def __init__(self, config, loss_weights, label_smoothing):
        super().__init__(config)
        
        self.label_smoothing = label_smoothing
        self.num_labels = config.num_labels
        self.loss_weights = nn.parameter.Parameter(torch.Tensor(loss_weights), requires_grad=False)
        self.swin = SwinModel(config)
        self.classifier = (
            nn.Linear(self.swin.num_features, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )
        self.post_init()
        
    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.swin(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(
                weight=self.loss_weights,
                label_smoothing=self.label_smoothing
            )
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


def set_envs_and_configs(local_rank):
    config = Config()
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    torch.manual_seed(config.SEED)
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"Environment variables, seeds and configs initialized at local rank: {local_rank}")
    return config


def get_dataloaders(config: Config, local_rank, world_size):
    transform_train = Compose([
        FancyPCA(
            alpha=0.1,
            p=0.2
        ),
        ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.3,
            hue=0.3,
            p=0.15
        ),
        Affine(
            scale=(0.8, 1.1),
            translate_percent=(0, 5),
            rotate=(-160, 160),
            shear=(-5, 5),
            fit_output=True,
            interpolation=cv2.INTER_CUBIC,
            p=0.3
        ),
        Perspective(
            scale=0.05,
            fit_output=True,
            interpolation=cv2.INTER_CUBIC,
            p=0.05
        ),
        Downscale(
            scale_min=0.25,
            scale_max=0.95,
            interpolation=cv2.INTER_CUBIC,
            p=0.1
        ),
        ChannelShuffle(p=0.05),
        Resize(
            height=384,
            width=384,
            interpolation=cv2.INTER_CUBIC,
            p=1.0
        ),
        Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2(p=1.0)  
    ])
    transform_val = Compose([
        Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2(p=1.0) 
    ])
    
    train_dataset = BinaryFaceDataset(
        folder = config.DATA_SUBSET_PATH_TRAIN, 
        transform = transform_train,
        sample = config.DATA_SUBSET_SIZE_TRAIN
    )
    val_dataset = BinaryFaceDataset(
        folder = config.DATA_SUBSET_PATH_VAL,
        transform = transform_val,
        sample = config.DATA_SUBSET_SIZE_VAL
    )
    print(f"Train and Val datasets initialized at local rank: {local_rank}")

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas = world_size,
        rank = local_rank,
        shuffle = config.DATA_SUBSET_SHUFFLE_TRAIN,
        seed = config.SEED
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas = world_size,
        rank = local_rank,
        shuffle = config.DATA_SUBSET_SHUFFLE_VAL,
        seed = config.SEED
    )
    print(f"Train and Val samplers initialized at local rank: {local_rank}")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler = train_sampler,
        batch_size = config.DATA_SUBSET_BATCH_SIZE_TRAIN,
        num_workers = config.DATALOADER_PROCESS_COUNT, 
        prefetch_factor = config.DATALOADER_PREFETCH_FACTOR,
        pin_memory = config.DATALOADER_PIN_MEMORY,
        drop_last = True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        sampler = val_sampler,
        batch_size = config.DATA_SUBSET_BATCH_SIZE_VAL,
        num_workers = config.DATALOADER_PROCESS_COUNT, 
        prefetch_factor = config.DATALOADER_PREFETCH_FACTOR,
        pin_memory = config.DATALOADER_PIN_MEMORY,
        drop_last = True
    )
    print(f"Train and Val dataloaders initialized at local rank: {local_rank}")
    return train_dataloader, val_dataloader


def get_model_optimizers_and_schedulers(config: Config, train_dataloader, local_rank, world_size):
    model = SwinForImageClassificationFixed.from_pretrained(
        config.SWIN_CHECKPOINT, 
        num_labels = config.NUM_CLASSES, 
        ignore_mismatched_sizes = True,
        loss_weights = config.LOSS_WEIGHTS,
        label_smoothing = config.LABEL_SMOOTHING
    )
    print(f"Model locally initialized at local rank: {local_rank}")

    embedding_opt_map_lookup = 'swin.embeddings'
    encoder_opt_map_lookup = [(i, f'swin.encoder.layers.{i}') for i in range(4)]
    layernorm_opt_map_lookup = 'swin.layernorm'
    classifier_opt_map_lookup = 'classifier'
    no_decay_list = ['bias', 'layernorm', 'dense']

    embedding_opt_map_decay = [{
        'params': [p for n, p in model.named_parameters() if (embedding_opt_map_lookup in n) and not any(type_nd in n for type_nd in no_decay_list)],
        'weight_decay_rate': config.WEIGHT_DECAY, 'lr': config.EMBEDDING_LR
    }]
    embedding_opt_map_no_decay = [{
        'params': [p for n, p in model.named_parameters() if (embedding_opt_map_lookup in n) and any(type_nd in n for type_nd in no_decay_list)],
        'weight_decay_rate': 0.0, 'lr': config.EMBEDDING_LR
    }]

    encoder_opt_map_decay = []
    encoder_opt_map_no_decay = []
    for num, l in encoder_opt_map_lookup:
        encoder_opt_map_decay.append(
            {
                'params': [p for n, p in model.named_parameters() if (l in n) and not any(type_nd in n for type_nd in no_decay_list)],
                'weight_decay_rate': config.WEIGHT_DECAY, 'lr': (config.ENCODER_BASE_LR * pow(config.ENCODER_LR_MULTIPLIER, 3 - num))
            }
        )
        encoder_opt_map_no_decay.append(
            {
                'params': [p for n, p in model.named_parameters() if (l in n) and any(type_nd in n for type_nd in no_decay_list)],
                'weight_decay_rate': 0.0, 'lr': (config.ENCODER_BASE_LR * pow(config.ENCODER_LR_MULTIPLIER, 3 - num))
            }
        )
        
    layernorm_opt_map_decay = [{
        'params': [p for n, p in model.named_parameters() if (layernorm_opt_map_lookup in n) and not any(type_nd in n for type_nd in no_decay_list)],
        'weight_decay_rate': config.WEIGHT_DECAY, 'lr': config.LAYERNORM_LR
    }]
    layernorm_opt_map_no_decay = [{
        'params': [p for n, p in model.named_parameters() if (layernorm_opt_map_lookup in n) and any(type_nd in n for type_nd in no_decay_list)],
        'weight_decay_rate': 0.0, 'lr': config.LAYERNORM_LR
    }]    

    classifier_opt_map_decay = [{
        'params': [p for n, p in model.named_parameters() if (classifier_opt_map_lookup in n) and not any(type_nd in n for type_nd in no_decay_list)],
        'weight_decay_rate': config.WEIGHT_DECAY, 'lr': config.CLASSIFIER_LR
    }]
    classifier_opt_map_no_decay = [{
        'params': [p for n, p in model.named_parameters() if (classifier_opt_map_lookup in n) and any(type_nd in n for type_nd in no_decay_list)],
        'weight_decay_rate': 0.0, 'lr': config.CLASSIFIER_LR
    }]

    collated_opt_maps = embedding_opt_map_decay + \
        embedding_opt_map_no_decay + \
        encoder_opt_map_decay + \
        encoder_opt_map_no_decay + \
        layernorm_opt_map_decay + \
        layernorm_opt_map_no_decay + \
        classifier_opt_map_decay + \
        classifier_opt_map_no_decay

    optimizer = Lamb(
        collated_opt_maps,
        lr=config.DEFAULT_LR,
        eps=config.EPSILON,
        weight_decay=config.WEIGHT_DECAY,
        clamp_value=config.CLIP_VALUE,
        adam=False,
        debias=True
    )

    num_warmup_steps = (len(train_dataloader.dataset) // config.DATA_SUBSET_BATCH_SIZE_TRAIN) * config.EPOCHS * config.WARMUP_PROP
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps if num_warmup_steps > config.MIN_WARMUP_STEPS else 0,
        num_cycles=config.NUM_CYCLES,
        num_training_steps=int(len(train_dataloader.dataset)/(config.DATA_SUBSET_BATCH_SIZE_TRAIN * world_size)) * config.EPOCHS
    )
    print(f"Optimizers and schedulers configured at local rank: {local_rank}")
    return model, optimizer, scheduler


def train_worker_main(local_rank):
    world_size = torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.distributed.init_process_group(
        backend="nccl",
        rank=local_rank,
        world_size=world_size
    )

    config: Config = set_envs_and_configs(local_rank)
    train_dataloader, val_dataloader = get_dataloaders(config, local_rank, world_size)
    model_local, optimizer, scheduler = get_model_optimizers_and_schedulers(config, train_dataloader, local_rank, world_size)
    model_local.to(device, non_blocking=False)
    model = nn.parallel.DistributedDataParallel(model_local, device_ids=[device], output_device=device)
    print(f"Model at local rank {local_rank} moved to device {next(model.parameters()).device} and DDP transformed")

    train_loss_record = []
    val_loss_record = []
    roc_auc_record = []
    average_precision_record = []
    exp_time = '_'.join(str(datetime.now()).split(' '))
    for epoch_i in trange(config.EPOCHS, desc='Epoch Loop', leave=True, disable=local_rank!=0, position=0):
        total_train_loss = torch.zeros([1], device=device)
        train_dataloader.sampler.set_epoch(epoch_i)
        model.train()

        for images, labels in tqdm(train_dataloader, desc='Train Loop', leave=False, disable=local_rank!=0, position=1):
            images_nv = images.to(device, non_blocking=True)
            labels_nv = labels.to(device, non_blocking=True)
            model.zero_grad(set_to_none=False)  
            loss, logits = model(
                pixel_values = images_nv,
                labels = labels_nv
            )
            loss.mean().backward()
            total_train_loss += loss.mean()
            optimizer.step()
            scheduler.step()

        dist.all_reduce(total_train_loss, op=dist.ReduceOp.SUM)
        avg_train_loss = total_train_loss.item() / (int(len(train_dataloader.dataset)/config.DATA_SUBSET_BATCH_SIZE_TRAIN) * config.DATA_SUBSET_BATCH_SIZE_TRAIN)
        if local_rank==0:
            train_loss_record.append(avg_train_loss)
            print(f"    Epoch {epoch_i+1} Train Loss: {avg_train_loss}") 
            output_dir = f'/home/ubuntu/Downloads/Models/Swin/{config.EXP_NAME}_{exp_time}/Epoch-'+str(epoch_i +1)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if hasattr(model, "module"):
                model.module.save_pretrained(output_dir)
            else:
                model.save_pretrained(output_dir)
            
        model.eval()
        total_eval_loss = torch.zeros([1], device=device)
        local_eval_logits = torch.zeros([0,2], device=device)
        local_eval_labels = torch.zeros([0], device=device)
        for images, labels in tqdm(val_dataloader, desc='Val Loop', leave=False, disable=local_rank!=0, position=1):
            images_nv = images.to(device, non_blocking=True)
            labels_nv = labels.to(device, non_blocking=True)
            with torch.no_grad():
                loss, logits = model(
                    pixel_values = images_nv,
                    labels = labels_nv
                )
            total_eval_loss += loss.mean()
            local_eval_logits = torch.cat([local_eval_logits, logits.detach()])
            local_eval_labels = torch.cat([local_eval_labels, labels_nv.detach()])
        
        dist.all_reduce(total_eval_loss, op=dist.ReduceOp.SUM)
        avg_val_loss = total_eval_loss.item() / (int(len(val_dataloader.dataset)/config.DATA_SUBSET_BATCH_SIZE_VAL) * config.DATA_SUBSET_BATCH_SIZE_VAL)
        global_eval_logits = [torch.zeros_like(local_eval_logits, device=device) for _ in range(world_size)]
        dist.all_gather(global_eval_logits, local_eval_logits)
        global_eval_labels = [torch.zeros_like(local_eval_labels, device=device) for _ in range(world_size)]
        dist.all_gather(global_eval_labels, local_eval_labels)
        if local_rank==0:
            val_loss_record.append(avg_val_loss)
            print(f"    Epoch {epoch_i+1} Val Loss: {avg_val_loss}")
            softmax_layer = nn.Softmax(dim = 1)
            all_logits = torch.vstack(global_eval_logits).cpu()
            all_labels = torch.hstack(global_eval_labels).cpu().numpy().astype(np.int8)
            test_probs = softmax_layer(all_logits).numpy()
            roc_auc = roc_auc_score(y_true=all_labels, y_score=test_probs[:, 1])
            roc_auc_record.append(roc_auc)
            print(f"    Epoch {epoch_i+1} ROC AUC Score: {roc_auc}") 
            prc_auc = average_precision_score(y_true=all_labels, y_score=test_probs[:, 1])
            average_precision_record.append(prc_auc)
            print(f"    Epoch {epoch_i+1} Average Precision: {prc_auc}") 
            print(f"=================================== Epoch {epoch_i+1} Done =====================================")

    if local_rank==0:
        stats_dict = {
            'Train_Loss': train_loss_record,
            'Val_Loss': val_loss_record,
            'ROC_AUC_Score': roc_auc_record,
            'Average_Precision': average_precision_record
        }
        with open(f'/home/ubuntu/Downloads/Models/Swin/{config.EXP_NAME}_{exp_time}/stats.json', 'w') as jfp:
            json.dump(stats_dict, jfp)
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train_worker_main, nprocs=torch.cuda.device_count())

