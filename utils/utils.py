import torch
import numpy as np
import random
import os
import wandb
import pandas as pd

def set_seed(config):
    torch.manual_seed(config.TRAIN.RANDOM_SEED)
    torch.cuda.manual_seed(config.TRAIN.RANDOM_SEED)
    torch.cuda.manual_seed_all(config.TRAIN.RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.TRAIN.RANDOM_SEED)
    random.seed(config.TRAIN.RANDOM_SEED)

def save_model(config, model):
    output_path = os.path.join(config.MODEL.SAVED_DIR, config.MODEL.MODEL_NAME)
    if not os.path.exists(config.MODEL.SAVED_DIR):
        os.makedirs(config.MODEL.SAVED_DIR)
    torch.save(model.state_dict(), output_path)
    
def init_wandb(config):
    """wandb 초기화 함수"""
    if config.WANDB.USE_SWEEP:
        with wandb.init() as run:
            # 기본 하이퍼파라미터 업데이트
            config.TRAIN.LR = wandb.config.get("TRAIN.LR", config.TRAIN.LR)
            config.TRAIN.OPTIMIZER.PARAMS.weight_decay = wandb.config.get(
                "TRAIN.OPTIMIZER.PARAMS.weight_decay", 
                config.TRAIN.OPTIMIZER.PARAMS.weight_decay
            )
            
            # Loss 설정 업데이트 
            config.TRAIN.LOSS.NAME = wandb.config.get("TRAIN.LOSS.NAME", config.TRAIN.LOSS.NAME)
            
            # Scheduler 설정 업데이트
            scheduler_name = wandb.config.get("TRAIN.SCHEDULER.NAME")
            config.TRAIN.SCHEDULER.NAME = scheduler_name
            
            # Scheduler별 파라미터 설정
            if scheduler_name == "CosineAnnealingLR":
                config.TRAIN.LOSS.PARAMS.eta_min = wandb.config.get("TRAIN.LOSS.PARAMS.eta_min")
            elif scheduler_name == "ExponentialLR":
                config.TRAIN.LOSS.PARAMS.gamma = wandb.config.get("TRAIN.LOSS.PARAMS.gamma")
            
            # SWA 설정 업데이트
            use_swa = wandb.config.get("TRAIN.SWA.USE_SWA", False)
            if use_swa:
                if not hasattr(config.TRAIN, 'SWA'):
                    config.TRAIN.SWA = type('', (), {})()  # 빈 객체 생성
                
                config.TRAIN.SWA.START = wandb.config.get("TRAIN.SWA.START")
                config.TRAIN.SWA.LR = wandb.config.get("TRAIN.SWA.LR")
                config.TRAIN.SWA.ANNEAL_EPOCHS = wandb.config.get("TRAIN.SWA.ANNEAL_EPOCHS")
                config.TRAIN.SWA.STRATEGY = wandb.config.get("TRAIN.SWA.STRATEGY")
            else:
                if hasattr(config.TRAIN, 'SWA'):
                    delattr(config.TRAIN, 'SWA')
            
            # Transform 설정 업데이트
            transforms_list = []
            
            # Resize는 기본 설정 유지
            transforms_list.extend([
                transform for transform in config.TRAIN.TRANSFORMS 
                if transform["NAME"] == "Resize"
            ])
            
            # Affine Transform
            if wandb.config.get("TRAIN.TRANSFORMS.use_affine", False):
                transforms_list.append({
                    "NAME": "Affine",
                    "PARAMS": {
                        "rotate": wandb.config.get("TRAIN.TRANSFORMS.affine_rotate")
                    }
                })
            
            # HorizontalFlip
            if wandb.config.get("TRAIN.TRANSFORMS.use_horizontal_flip", False):
                transforms_list.append({
                    "NAME": "HorizontalFlip",
                    "PARAMS": {}
                })
            
            # config의 transforms 리스트 업데이트
            config.TRAIN.TRANSFORMS = transforms_list
            
            # wandb 실험 설정
            wandb.run.name = f"{config.WANDB.RUN_NAME}_{wandb.run.id}"
            wandb.run.notes = config.WANDB.NOTES
            wandb.run.tags = config.WANDB.TAGS
            wandb.config.update(config.WANDB.CONFIGS)
    else:
        # 일반 모드로 실행될 때
        run = wandb.init(
            project=config.WANDB.PROJECT_NAME,
            entity=config.WANDB.ENTITY, 
            name=config.WANDB.RUN_NAME, 
            notes=config.WANDB.NOTES, 
            tags=config.WANDB.TAGS, 
            config=config.WANDB.CONFIGS
        )


def wandb_model_log(config):
    model_path = os.path.join(config.MODEL.SAVED_DIR, config.MODEL.MODEL_NAME)
    wandb.save(model_path)
    artifact = wandb.Artifact(name=f"{config.MODEL.MODEL_NAME}", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    
def save_csv(config, rles, mode, epoch=None):
    mode_config = getattr(config, mode.upper())
    os.makedirs(mode_config.OUTPUT_DIR, exist_ok=True)
    csv_name = f'epoch{epoch}_' + mode_config.CSV_NAME if epoch is not None else mode_config.CSV_NAME
    output_path = os.path.join(mode_config.OUTPUT_DIR, csv_name)
    df = pd.DataFrame({
        "image_name": rles['image_names'],
        "class": rles['classes'],
        "rle": rles['rles'],
    })
    df.to_csv(output_path, index=False)
    print(f"{mode} results saved to {output_path}")