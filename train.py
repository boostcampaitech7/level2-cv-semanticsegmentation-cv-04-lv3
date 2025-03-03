import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
#import torchvision.models as models
from tqdm import tqdm
import os

from utils.utils import set_seed, save_model, wandb_model_log, save_csv
from utils.metrics import dice_coef, encode_mask_to_rle
from utils.optimizer import get_optimizer
from utils.scheduler import get_scheduler
from utils.loss import get_loss
from data.dataset import XRayDataset
from data.augmentation import DataTransforms
import models

from config.config import Config

from torch.cuda.amp import autocast, GradScaler

import wandb
import argparse

from torch.optim.swa_utils import AveragedModel, SWALR

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    return parser.parse_args()


def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    result_rles = {'image_names': [], 'classes': [], 'rles': []}
    
    with torch.no_grad():
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
            outputs = model(images)
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr)

            
            dice = dice_coef(outputs, masks)
            dices.append(dice)

            # save validation outputs (rles) to 'result_rles' 
            dataset = data_loader.dataset
            batch_filenames = [dataset.filenames[i] for i in range(step * data_loader.batch_size, min((step + 1) * data_loader.batch_size, len(dataset.filenames)))]
    
            for output, image_name in zip(outputs, batch_filenames):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm.cpu())
                    result_rles['rles'].append(rle)
                    result_rles['classes'].append(data_loader.dataset.IND2CLASS[c])
                    result_rles['image_names'].append(os.path.basename(image_name))
            
        print('val total loss: ', (total_loss/cnt))
        wandb.log({"val/loss": total_loss/cnt})
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    
    dice_dict = {f"val/{c}": d.item() for c, d in zip(config.DATA.CLASSES, dices_per_class)}
    
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(config.DATA.CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    print('avg_dice: ', avg_dice)
    
    return avg_dice, dice_dict, result_rles

def train(model, data_loader, val_loader, criterion, optimizer, scheduler):
    run = wandb.init(
        project=config.WANDB.PROJECT_NAME,
        entity=config.WANDB.ENTITY, 
        name=config.WANDB.RUN_NAME, 
        notes=config.WANDB.NOTES, 
        tags=config.WANDB.TAGS, 
        config=config.WANDB.CONFIGS
    )
    wandb.watch(model, criterion, log="all", log_freq=config.WANDB.WATCH_STEP*len(data_loader))
    
    print(f'Start training..')
    
    best_dice, best_dice_epoch = 0., 0.
    best_rles = None
    scaler = GradScaler() if config.TRAIN.FP16 else None

    if hasattr(config.TRAIN, 'SWA'):
        swa_model = AveragedModel(model).cuda()
        swa_start = config.TRAIN.SWA.START
        swa_scheduler = SWALR(optimizer=optimizer, 
                              swa_lr=config.TRAIN.SWA.LR, 
                              anneal_epochs=config.TRAIN.SWA.ANNEAL_EPOCHS,
                              anneal_strategy=config.TRAIN.SWA.STRATEGY)

    
    if hasattr(config.TRAIN, 'ACCUMULATION_STEPS'):
        accumulation_steps = config.TRAIN.ACCUMULATION_STEPS
    else:
        accumulation_steps = 1

    for epoch in range(config.TRAIN.EPOCHS):
        model.train()
        current_lr = scheduler.get_last_lr()[0]  # 첫 번째 학습률을 가져옵니다.
        print(f'Epoch [{epoch+1}/{config.TRAIN.EPOCHS}] | Learning Rate: {current_lr}') 

        optimizer.zero_grad()

        for step, (images, masks) in enumerate(data_loader):            
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()

            if config.TRAIN.FP16:
                # FP16 사용 시
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()
                if (step+1) % accumulation_steps == 0 or (step + 1) == len(data_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            else:
                # FP32 사용 시
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
        
            
            # step 주기에 따라 loss를 출력합니다.
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{config.TRAIN.EPOCHS}], '
                    f'Step [{step+1}/{len(data_loader)}], '
                    f'Loss: {round(loss.item() * accumulation_steps,4)}'
                )
                wandb.log({"train/loss": round(loss.item() * accumulation_steps,4), "lr": current_lr, "epoch": epoch+1})

        if hasattr(config.TRAIN, 'SWA'):
            if epoch+1 >= swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()
        else:    
            scheduler.step()
        
        
        # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
        if (epoch + 1) % config.TRAIN.VAL_EVERY == 0:
            dice, class_dices, rles = validation(epoch + 1, model, val_loader, criterion)
            wandb.log({"val/avg_dice": dice, **class_dices})
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {config.MODEL.SAVED_DIR}")
                best_dice, best_dice_epoch = dice, epoch+1
                best_rles = rles
                save_model(config, model)
                wandb_model_log(config)
    
    if best_rles:
        save_csv(config, best_rles, mode='TRAIN', epoch=best_dice_epoch)

    if hasattr(config.TRAIN, 'SWA'):
        torch.optim.swa_utils.update_bn(data_loader, swa_model, device="cuda")

        output_path = os.path.join(config.MODEL.SAVED_DIR, config.TRAIN.SWA.MODEL_NAME)
        if not os.path.exists(config.MODEL.SAVED_DIR):
            os.makedirs(config.MODEL.SAVED_DIR)
        torch.save(swa_model.state_dict(), output_path)
        
    wandb.finish()
    
def main():
    data_transforms = DataTransforms(config)

    tf_train = data_transforms.get_transforms("train")
    tf_valid = data_transforms.get_transforms("valid")
    
    train_dataset = XRayDataset(is_train=True, transforms=tf_train, config=config)
    valid_dataset = XRayDataset(is_train=False, transforms=tf_valid, config=config)

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )
    
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=4,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )


    # model 불러오기
    model_class = getattr(models, config.MODEL.TYPE)  # models에서 모델 클래스 가져오기
    model = model_class(config).get_model()

    if config.TRAIN.PRETRAIN.USE_PRETRAINED:
        state_dict = torch.load(config.TRAIN.PRETRAIN.MODEL_PATH)
        model.load_state_dict(state_dict)
        print(f"Pretrained model load from {config.TRAIN.PRETRAIN.MODEL_PATH}")

    criterion = get_loss(config)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    # 학습 시작
    set_seed(config)
    train(model, train_loader, valid_loader, criterion, optimizer, scheduler)

if __name__ == '__main__':
    args = parse_args()

    config = Config(args.config)
    
    main()
