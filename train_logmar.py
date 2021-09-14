import os, json
import time
import random
import argparse
import numpy as np
from numpy.lib.twodim_base import diag
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import GradualWarmupSchedulerV2
import apex
from apex import amp
from dataset import get_df, get_transforms, DatasetFromCsv
from models import Effnet_Custom 

from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


from SoftLabelLoss import cross_entropy_with_probs

import OtherMetrics


def _parse_comma_sep(s):
    if s is None or s.lower() == 'none' or s == '':
        return None
    return s.split(',')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--image-csv', type=str, default=None)
    parser.add_argument('--label-upweigh', type=str, default=None)
    parser.add_argument('--data-folder', type=int)
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=32)
    parser.add_argument('--init-lr', type=float, default=3e-5)
    parser.add_argument('--out-dim', type=int, default=9)
    parser.add_argument('--n-epochs', type=int, default=15)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--use-meta', action='store_true') # ! will be override later. 
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--fold', type=str, default='1,2,3,4')
    parser.add_argument('--n-meta-dim', type=str, default='512,128') # ! input and output of meta-features
    
    # ! added
    parser.add_argument('--linear-loss', action='store_true', default=False)
    parser.add_argument('--fillna0', action='store_true', default=False)
    parser.add_argument('--meta-features', type=_parse_comma_sep, default=None, help='column names a,b,c') # ! column names
    parser.add_argument('--seed', type=float, default=0)
    parser.add_argument('--n-test', type=int, default=1, help='how many times do we flip images, 1=>no_flip, max=8')
    parser.add_argument('--scheduler-scaler', type=float, default=10)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--no-scheduler', action='store_true', default=False)
    parser.add_argument('--our-data', type=str, default=None)
    parser.add_argument('--weighted-loss', type=str, default=None) # string input to have many weights on each kind of label
    parser.add_argument('--weighted-loss-ext', type=float, default=1) ## decrease external weights
    parser.add_argument('--img-map-file', type=str, default='train.csv')
    parser.add_argument('--loaded-model', type=str, default=None)
    parser.add_argument('--soft-label', action='store_true', default=False)
    parser.add_argument('--new-label', type=str, default=None)

    args = parser.parse_args() # ! safer 
    return args


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, optimizer, criterion, args=None):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target, target_soft, _, _) in bar: # added path name and original resize image (only needed during attribution)

        optimizer.zero_grad()
        
        if args.use_meta:
            data, meta = data
            data, meta, target = data.to(device), meta.to(device), target.to(device)
            logits = model(data, meta)
        else:
            data, target = data.to(device), target.to(device)
            logits = model(data)        
        
        loss = criterion(logits, target.unsqueeze(1))
                
        if not args.use_amp:
            loss.backward()
        else:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        if args.image_size in [896,576]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        #
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))

    train_loss = np.mean(train_loss)
    return train_loss


def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2, 3)
    if I % 4 == 0: # ! return original if I = 0
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)


def val_epoch(model, loader, n_test=1, is_ext=None, get_output=False, criterion=None, args=None):

    model.eval()
    val_loss = []
    LOGITS = []
    TARGETS = []
    with torch.no_grad():
        for (data, target, target_soft, _, _) in tqdm(loader): # added path name, and original resize
            
            if args.use_meta:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I), meta)
                    logits += l
            else:
                data, target = data.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I))
                    logits += l

            # average over all the augmentation of test data
            logits /= n_test

            LOGITS.append(logits.detach().cpu())
            TARGETS.append(target.detach().cpu()) # ! because we keep original @target, we can't override @target with @target_soft until later

            loss = criterion(logits, target.unsqueeze(1))
            
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy() # ! 2D array
    TARGETS = torch.cat(TARGETS).numpy() # ! 1D array 

    if get_output:
        return LOGITS
    else:
        reg = LinearRegression().fit(LOGITS, TARGETS) # reg = LinearRegression().fit(X, y) # ! X is 2D, y can be 1D
        r2score = reg.score(LOGITS, TARGETS)
        LOGITS = LOGITS.flatten() 
        TARGETS = TARGETS.flatten()
        corr, _ = pearsonr(LOGITS, TARGETS) 
        return val_loss, corr, r2score


def run(fold, df, meta_features, n_meta_features, transforms_train, transforms_val, criterion):

    if len(set(df['fold'].values)) == 6: # ! otherwise, we have a specialized test set, so we don't have to filter
        print ('fold 5 is designed as the test set, len before remove fold id=5 {}'.format(df.shape[0]))
        df = df[df['fold'] != 5]
        print ('len after remove fold id=5 {}'.format(df.shape[0]))
        
    if args.DEBUG:
        args.n_epochs = 5
        df_train = df[df['fold'] != fold].sample(args.batch_size * 5)
        df_valid = df[df['fold'] == fold].sample(args.batch_size * 5)
    else:
        df_train = df[df['fold'] != fold] # ! take out a fold and keep it as valid
        df_valid = df[df['fold'] == fold]
        # print ('df input size {}'.format(df.shape[0]))
        # print ('df_train input size after remove fold {} {}'.format(fold,df_train.shape[0]))
        # print ('df_valid input size after remove fold {} {}'.format(fold,df_valid.shape[0]))

    # ! check label count
    print ( 'final df train labels count {}'.format ( df_train.groupby('label').count() ) )
    print ( 'final df valid labels count {}'.format ( df_valid.groupby('label').count() ) )

    dataset_train = DatasetFromCsv(df_train, 'train', meta_features, transform=transforms_train, soft_label=args.soft_label, linear_loss=args.linear_loss)
    dataset_valid = DatasetFromCsv(df_valid, 'valid', meta_features, transform=transforms_val, soft_label=args.soft_label, linear_loss=args.linear_loss)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, sampler=RandomSampler(dataset_train), num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers)
    
    print('train and dev data size {} , {}'.format(len(dataset_train), len(dataset_valid)))

    model = ModelClass(
        args.enet_type,
        n_meta_features=n_meta_features,
        n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
        out_dim=args.out_dim,
        pretrained=True, 
        args=args
    )

    model = model.to(device)

    # ! loading in a model
    if args.loaded_model is not None: 
        print ('\nloading {}\n'.format(args.loaded_model))
        
        try:  # single GPU model_file
            if args.CUDA_VISIBLE_DEVICES is None:
                state_dict = torch.load(args.loaded_model, map_location=torch.device('cpu'))
            else: 
                state_dict = torch.load(args.loaded_model)
      
        except:  # multi GPU model_file
            if args.CUDA_VISIBLE_DEVICES is None:
                state_dict = torch.load(args.loaded_model, map_location=torch.device('cpu'))
            else: 
                state_dict = torch.load(args.loaded_model)
                
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}

        # ! load model after get @state_dict
        if args.use_meta or args.linear_loss: 
            print ('use meta feature, will remove last fc layer from 100k oct pretrained.')
            temp_ = [] 
            for key in state_dict.keys():
                if 'myfc' in key: # ! 100k oct doesn't have meta-features
                    temp_.append(key)
            # print
            print ('not laod {}'.format(temp_))
            for key in temp_: 
                del state_dict[key] 
    
        model.load_state_dict(state_dict, strict=False)

    # ! send to multiple gpus... only works well if we have model.forward, don't change forward func.
    if DP:
        model = apex.parallel.convert_syncbn_model(model)

    r2score_max = 0.
    model_file  = os.path.join(args.model_dir, f'{args.kernel_type}_best_all_fold{fold}.pth')
    model_file_final = os.path.join(args.model_dir, f'{args.kernel_type}_final_fold{fold}.pth')

    # ! our own label, we may add normal ?? ... but we are not right now.
    # r2score_max = 0. 
    model_file_our = os.path.join(args.model_dir, f'{args.kernel_type}_ourlabel_fold{fold}.pth')
    
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    if args.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    if DP:
        model = nn.DataParallel(model)

    if not args.no_scheduler: 
        # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs - 1)
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
        scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=args.scheduler_scaler, total_epoch=1, after_scheduler=scheduler_cosine)

    best_epoch = 0 # ! early stop 
    
    for epoch in range(1, args.n_epochs + 1):
        print(time.ctime(), f'Fold {fold}, Epoch {epoch}')
		# scheduler_warmup.step(epoch - 1)

        train_loss = train_epoch(model, train_loader, optimizer, criterion=criterion, args=args)
        val_loss, corr, r2score = val_epoch(model, valid_loader, n_test=args.n_test, is_ext=df_valid['is_ext'].values, criterion=criterion, get_output=False, args=args) 

        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}, corr: {(corr):.4f}, r2score {(r2score):.6f}'
        print(content)
        with open(os.path.join(args.log_dir, f'log_{args.kernel_type}.txt'), 'a') as appender:
            appender.write(content + '\n')

        if not args.no_scheduler: 
            scheduler_warmup.step()    
            if epoch==2: scheduler_warmup.step() # bug workaround   
                
        # if r2score > r2score_max: # ! save best model on all labels
        #     print('r2score_max ({:.6f} --> {:.6f}). Saving model ...'.format(r2score_max, r2score))
        #     torch.save(model.state_dict(), model_file) ## @model is the same model in both cases, just eval them separately
        #     r2score_max = r2score
        #     best_epoch = epoch 
        
        # ! save model best for our data
        if r2score > r2score_max:
            print('r2score_max ({:.6f} --> {:.6f}). Saving model ...'.format(r2score_max, r2score))
            torch.save(model.state_dict(), model_file_our)
            r2score_max = r2score
            best_epoch = epoch 
            
        # ! early stop based on acc. for our data
        if epoch - best_epoch > 10 : 
            print ('break early')
            print (epoch - best_epoch)
            break 
        
    # ! end loop
    torch.save(model.state_dict(), model_file_final)


def main():

    # ! meta data (i.e. left/right eye, machine etc.)
    if args.meta_features is not None: 
        if args.new_label is not None: 
            args.meta_features = [i for i in args.meta_features if i not in [args.new_label]] # ! remove just in case. 
    #
    n_meta_features = 0 if args.meta_features is None else len(args.meta_features)
    args.use_meta = False if args.meta_features is None else True 
    
    df, diagnosis2idx, _ = get_df(args.image_csv, soft_label=args.soft_label, args=args)
   
    # ! linear regression loss 
    criterion = nn.SmoothL1Loss()

    # ! data aug transformation
    transforms_train, transforms_val, _ = get_transforms(args.image_size) # don't need 3rd transform resize 

    folds = [int(i) for i in args.fold.split(',')]
    print ('\nfolds {}\n'.format(folds))
    for fold in folds: # ! run many folds
        run(fold, df, args.meta_features, n_meta_features, transforms_train, transforms_val, criterion)
        

if __name__ == '__main__':

    args = parse_args()
    with open(os.path.join(args.log_dir,'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    if args.enet_type == 'resnest101':
        ModelClass = Resnest_Custom
    elif args.enet_type == 'seresnext101':
        ModelClass = Seresnext_Custom
    elif 'efficientnet' in args.enet_type:
        ModelClass = Effnet_Custom
    else:
        raise NotImplementedError()

    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    set_seed(seed=args.seed) # ! set a seed, default to 0

    device = torch.device('cuda')
   
    main()

