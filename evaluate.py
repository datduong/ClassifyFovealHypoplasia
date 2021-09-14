import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import cv2
import PIL.Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
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
from models import Effnet_Custom # , Resnest_Custom, Seresnext_Custom
from train import get_trans

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel

from copy import deepcopy 

import OtherMetrics
from SeeAttribution import GetAttribution

from SoftLabelLoss import cross_entropy_with_probs

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
    parser.add_argument('--out-dim', type=int, default=9)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--oof-dir', type=str, default='./oofs')
    parser.add_argument('--eval', type=str, choices=['best', 'best_20', 'final','best_all','ourlabel'], default="ourlabel") # "best")
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default=None) # '0'
    parser.add_argument('--n-meta-dim', type=str, default='512,128')
    parser.add_argument('--fold', type=str, default='0,1,2,3,4')
    
    # ! added
    parser.add_argument('--linear-loss', action='store_true', default=False)
    parser.add_argument('--fillna0', action='store_true', default=False)
    parser.add_argument('--meta-features', type=_parse_comma_sep, default=None, help='column names a,b,c') # ! column names
    parser.add_argument('--dropout', type=float, default=0.5) # doesn't get used
    parser.add_argument('--n-test', type=int, default=1, help='how many times do we flip images, 1=>no_flip, max=8')
    parser.add_argument('--attribution_keyword', type=str, default=None) 
    parser.add_argument('--outlier_perc', type=int, default=10, help='show fraction of high contributing pixel, default 10%')
    parser.add_argument('--img-map-file', type=str, default='train.csv')
    parser.add_argument('--do_test', action='store_true', default=False)
    parser.add_argument('--soft-label', action='store_true', default=False)
    parser.add_argument('--new-label', type=str, default=None)
    parser.add_argument('--attribution_model', type=str, default='integrated_gradient')
    parser.add_argument('--noise_tunnel', action='store_true', default=False)

    args = parser.parse_args()
    return args



def val_epoch(model, loader, our_label_index, diagnosis2idx, n_test=1, get_output=True, fold=None, args=None):

    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []

    if args.attribution_keyword is not None: # ! do attribution
        n_test = 1 # ! test original image, not flipping
        print ('\nwill do attribution_model\n')
        if args.attribution_model == 'Occlusion': 
            attribution_model = Occlusion(model)
        else: 
            attribution_model = IntegratedGradients(model) # send back to cpu, cuda takes up too much space
            if args.noise_tunnel: 
                attribution_model = NoiseTunnel(attribution_model)

    with torch.no_grad():
        for (data, target, target_soft, path, data_resize) in tqdm(loader): # @path is needed to select our labels

            if args.attribution_keyword is not None: 
                our_label_index = [path.index(j) for j in path if args.attribution_keyword in j] # get only NF1 or HMI or etc...
                if len (our_label_index) == 0 :
                    continue # skip
                    
            if args.use_meta:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I), meta)
                    logits += l
                    probs += l.softmax(1)
            else:
                data, target = data.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I))
                    logits += l
                    probs += l.softmax(1)
            logits /= n_test
            probs /= n_test

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu()) # ! @PROBS is shape=(1, obs, labelsize)
            TARGETS.append(target.detach().cpu())

            if args.soft_label: 
                loss = criterion(logits, target_soft.to(device))
            else:
                loss = criterion(logits, target)
            
            val_loss.append(loss.detach().cpu().numpy())

            # ! do attribution here. call IntegratedGradient, or some other approaches
            if args.attribution_keyword is not None: 
                our_label_index = [path.index(j) for j in path if args.attribution_keyword in j.split('/')[-1]] # 
                if len (our_label_index) > 0 :
                    temp = GetAttribution.GetAttributionPlot (  data[our_label_index].detach().cpu(), 
                                                                probs[our_label_index].detach().cpu(), 
                                                                np.array(path)[our_label_index], 
                                                                data_resize[our_label_index], 
                                                                attribution_model, 
                                                                fold=fold,
                                                                true_label_index=target[our_label_index].detach().cpu(),
                                                                args=args)

    # ! end eval loop
    if args.attribution_keyword is not None: 
        exit() # ! just do attribution
        
    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy() # ! @PROBS is shape=(1, 11708, 11)
    TARGETS = torch.cat(TARGETS).numpy()

    # ! compute acc for this fold. 
    
    acc = (PROBS.argmax(1) == TARGETS).mean() * 100.

    for key,idx in diagnosis2idx.items(): # ! AUC
        if idx in our_label_index: 
            auc = roc_auc_score((TARGETS == idx).astype(float), PROBS[:, idx]) 
            print (time.ctime() + ' ' + f'Fold {fold}, {key} auc: {auc:.5f}')

    # ! weighted accuracy
    bal_acc = OtherMetrics.compute_balanced_accuracy_score(PROBS, TARGETS)

    # ! global confusion matrix
    OtherMetrics.plot_confusion_matrix( PROBS, TARGETS, diagnosis2idx, os.path.join(args.log_dir,'confusion_matrix_fold'+str(fold) ), our_label_index )
    
    return LOGITS, PROBS, val_loss, acc, bal_acc



def main():

    # ! meta data (i.e. left/right eye, machine etc.)
    if args.meta_features is not None: 
        if args.new_label is not None: 
            args.meta_features = [i for i in args.meta_features if i not in [args.new_label]] # ! remove just in case. 
    #
    n_meta_features = 0 if args.meta_features is None else len(args.meta_features)
    args.use_meta = False if args.meta_features is None else True 
    
    print (args)

    if args.use_meta: 
        print ('n_meta_features :', n_meta_features)
    
    df, diagnosis2idx, our_label_index = get_df(args.image_csv, soft_label=args.soft_label, args=args)

    _, transforms_val, transforms_resize = get_transforms(args.image_size)

    ## see our input 
    print ('our_label_index {}'.format(our_label_index))
   
    LOGITS = []
    PROBS = []
    for fold in [int(i) for i in args.fold.split(',')]:

        if (args.do_test) and (len(set(df['fold'].values))==6) : 
            print ('\ntesting on fold id=5 using data trained without fold {}\n'.format(fold))
            df_valid = df[df['fold'] == 5] # ! eval on our own test set
        elif args.do_test: 
            df_valid = df # ! we have a separated test set. 
        else: 
            df_valid = df[df['fold'] == fold] # ! eval on the left-out fold

        print ('eval data size {}'.format(df_valid.shape))

        dataset_valid = DatasetFromCsv(df_valid, 'valid', args.meta_features, transform=transforms_val, transform_resize=transforms_resize, soft_label=args.soft_label)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

        print ('len of valid pytorch dataset {}'.format(len(dataset_valid)))
        
        # ! load model         
        model_file = os.path.join(args.model_dir, f'{args.kernel_type}_{args.eval}_fold{fold}.pth')
        print ('\nmodel_file {}\n'.format(model_file))
        
        model = ModelClass(
            args.enet_type,
            n_meta_features=n_meta_features,
            n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
            out_dim=args.out_dim,
            pretrained=True, 
            args=args
        )
    
        model = model.to(device)

        try:  # single GPU model_file
            if args.CUDA_VISIBLE_DEVICES is None:
                model.load_state_dict(torch.load(model_file), strict=True, map_location=torch.device('cpu')) # ! avoid error in loading model trained on GPU
            else: 
                model.load_state_dict(torch.load(model_file), strict=True) 
        except:  # multi GPU model_file
            if args.CUDA_VISIBLE_DEVICES is None:
                state_dict = torch.load(model_file, map_location=torch.device('cpu'))
            else: 
                state_dict = torch.load(model_file)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            model.load_state_dict(state_dict, strict=True)
        
        if DP : # len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
            model = torch.nn.DataParallel(model)

        model.eval()

        if args.use_meta:
            print (model)

        this_LOGITS, this_PROBS, val_loss, acc, bal_acc = val_epoch(model, valid_loader, our_label_index, diagnosis2idx, n_test=args.n_test, get_output=True, fold=fold, args=args)
        LOGITS.append(this_LOGITS)
        PROBS.append(this_PROBS)

        # print 
        content = time.ctime() + ' ' + f'Fold {fold}, valid loss: {(val_loss):.5f}, acc: {(acc):.4f}, bal_acc {(bal_acc):.6f}'
        print(content)
        with open(os.path.join(args.log_dir, f'log_{args.kernel_type}_eval.txt'), 'a') as appender:
            appender.write(content + '\n')

        # ! merge data frame
        print ('PROB output size {}'.format(PROBS[0].shape))
        prob_df = pd.DataFrame( PROBS[0], columns=[str(i) for i in np.arange(args.out_dim)]) # @PROBS is shape=(1, obs, labelsize)
        prob_df = prob_df.reset_index(drop=True) ## has to do this to concat right
        df_valid_temp = df_valid.copy()
        df_valid_temp = df_valid_temp.reset_index(drop=True)
        print ('dim df_valid_temp {} and prob_df {}'.format(df_valid_temp.shape,prob_df.shape))
        assert df_valid_temp.shape[0] == prob_df.shape[0]
        df_valid_prob = pd.concat([df_valid_temp, prob_df], axis=1) # ! just append col wise
        log_file_name = 'eval_fold_'+str(fold)+'.csv'
        if args.do_test:
            log_file_name = 'test_on_fold_5_from_fold'+str(fold)+'.csv' # ! special fold 0, which is our test set

        # ! compute average score  
        try: 
            df_valid_prob['average_score'] = np.matmul (df_valid_prob[['0','1','2','3']].to_numpy() , np.reshape([1,2,3,4],(4,1))).flatten().tolist()
        except: 
            pass
        
        df_valid_prob.to_csv(os.path.join(args.log_dir, log_file_name),index=False)

    # end folds  



if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.oof_dir, exist_ok=True)

    # if args.enet_type == 'resnest101':
    #     ModelClass = Resnest_Custom
    # elif args.enet_type == 'seresnext101':
    #     ModelClass = Seresnext_Custom
    if 'efficientnet' in args.enet_type:
        ModelClass = Effnet_Custom
    else:
        raise NotImplementedError()

    if args.CUDA_VISIBLE_DEVICES is not None: 
        os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
        DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1
        device = torch.device('cuda')
    else: 
        DP = False  
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not args.soft_label:     
        criterion = nn.CrossEntropyLoss()
    else: 
        criterion = cross_entropy_with_probs

    main()

