import sys
sys.path.append('./util')
sys.path.append('./model')
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from prepro_data import Batch_generator
from evaluation import cal_cc_score,cal_sim_score,cal_kld_score,cal_spearman_corr,cal_auc
from model import Model
from model_att import Model_Att
from model_add import Model_Add
from torch.autograd import Variable
import numpy as np
import cv2
import argparse
import os

parser = argparse.ArgumentParser(description='Saliency Prediction for VQA attention')
parser.add_argument('--mode', type=str, default='training', help='Selecting running mode (default: train)')
parser.add_argument('--model', type=str, default='normal', help='Selecting model to be trained (default: normal concatenation)')
parser.add_argument('--weights',type=str, default=None, help='Trained model to be loaded (default: None)')
parser.add_argument('--save_map',type=bool, default=False, help='Saving result maps and ground truth (default: False)')
parser.add_argument('--save_dir',type=str, default='./res', help='Directory to save maps (default: ./res)')
args = parser.parse_args()

def init_metrics(metrics):
    metric = dict()
    for k in metrics:
        metric[k] = 0
    metric['count'] = 0

    return metric


def logsoftmax_cross_entropy(input_,target):
    "original cross entropy loss only works with one-hot vetcor, write an own one"
    epsilon = 2.22044604925031308e-16
    input_ = input_.view(input_.size(0), -1)
    target = target.view(target.size(0), -1)
    softmax = torch.exp(input_-torch.max(input_).expand_as(input_))/torch.sum(torch.exp(input_-torch.max(input_).expand_as(input_))).expand_as(input_)
    loss = -torch.mean(target*torch.log(torch.clamp(softmax,min=epsilon,max=1)))
    loss = torch.mean(loss)
    return loss

def get_map(input_):
    input_ = input_.view(input_.size(0), -1)
    input_ = input_.cpu().numpy()
    softmax = np.exp(input_-np.max(input_))/np.sum(np.exp(input_-np.max(input_)))

    return softmax

def log(file_name,msg):
    log_file = open(file_name,"a")
    log_file.write(msg+'\n')
    log_file.close()

def adjust_learning_rate(optimizer, epoch):
    "adatively adjust lr based on epoch"
    if epoch < 10:
        lr = 2.5e-3
    elif epoch <= 20:
        lr = 2.5e-3 * (10 ** (float(epoch-10) / 10))
    elif epoch<=40:
        lr = 2.5e-2 * (0.1 ** (float(epoch-20) / 20))
    else:
        lr = 2.5e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def process_map(sal_map,fix_map,img_rows,img_cols,save_map=False,save_dir=None,idx=None):
    sal_map = get_map(sal_map.data)
    sal_map = sal_map.reshape((img_rows/32,img_cols/32))
    fix_map = get_map(fix_map.data)
    fix_map = fix_map.reshape((img_rows/32,img_cols/32))

    sal_map = (sal_map-np.min(sal_map))/(np.max(sal_map)-np.min(sal_map))
    fix_map = (fix_map-np.min(fix_map))/(np.max(fix_map)-np.min(fix_map))

    if save_map == True:
        cv2.imwrite(os.path.join(save_dir,str(idx)+'_pred.jpg'),sal_map*255)
        cv2.imwrite(os.path.join(save_dir,str(idx)+'_gt.jpg'),fix_map*255)

    sal_map = cv2.resize(sal_map,(img_cols,img_rows),interpolation = cv2.INTER_LINEAR)
    fix_map = cv2.resize(fix_map,(img_cols,img_rows),interpolation = cv2.INTER_LINEAR)

    return sal_map,fix_map


def train():
    #parameters
    nb_epoch = 100
    img_rows,img_cols = 480, 640
    batch_size = 1
    imgDir = '/home/luoyan/project/dataset/mscoco/images/train2014'
    mapDir = '/home/chenshi/VQA/dataset/vqahat/vqahat_train'
    annFile = '/home/chenshi/VQA/dataset/VQA_1/annotation/mscoco_train2014_annotations.json'
    quesFile = '/home/chenshi/VQA/dataset/VQA_1/question/OpenEnded_mscoco_train2014_questions.json'
    embeddingDir = '/home/chenshi/VQA/dataset/word_embedding/GoogleNews-vectors-negative300.bin'

    #initialize data loader
    DataLoader = Batch_generator(img_rows,img_cols,batch_size,imgDir,mapDir,annFile,quesFile,embeddingDir,val_split=0.2)

    #defining model and optimizer
    if args.model == 'normal':
        model = Model()
        model.load_state_dict(torch.load('pretrained.pth')) #loading pretrained weights
    elif args.model == 'attention':
        model = Model_Att()
        model.load_state_dict(torch.load('pretrained_att.pth'))
    elif args.model == 'addition':
        model = Model_Add()
        model.load_state_dict(torch.load('pretrained_add.pth')) #loading pretrained weights
    else:
        assert 0, 'Invalid model selected'
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=2.5e-4, weight_decay=5e-4,momentum=0.9,nesterov=True) #0.001

    #private function for training and evaluation
    def train(epoch):
        model.train()

        for batch_idx in xrange(len(DataLoader.training_idx)):
            img, que, target,flag = DataLoader.get_batch(batch_idx,mode='train')
            if flag == 0:
                continue
            img, que, target = torch.from_numpy(img), torch.from_numpy(que), torch.from_numpy(target)
            img, que, target = Variable(img), Variable(que), Variable(target)
            img, que, target = img.cuda(), que.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(img,que)
            loss=logsoftmax_cross_entropy(output,target)
            # loss = F.cross_entropy(output,target)
            loss.backward()
            optimizer.step()
            if batch_idx % 1000 == 0 or batch_idx == DataLoader.iter_num-1:
                msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(img), len(DataLoader.training_idx),
                    100. * batch_idx * len(img)/ len(DataLoader.training_idx), loss.data[0])
                print msg
                log('training_loss_v2.log',msg)

    def test(epoch):
        model.eval()
        total_loss = []
        for batch_idx in xrange(len(DataLoader.val_idx)):
            img, que, target,flag = DataLoader.get_batch(batch_idx,mode='val')
            if flag == 0 :
                continue
            img, que, target = torch.from_numpy(img), torch.from_numpy(que), torch.from_numpy(target)
            img, que, target = Variable(img,volatile=True), Variable(que,volatile=True), Variable(target,volatile=True)
            img, que, target = img.cuda(), que.cuda(), target.cuda()
            output = model(img,que)
            # loss=criterion(output,target.long())
            loss = logsoftmax_cross_entropy(output,target)
            total_loss.append(loss.data[0])
        total_loss = np.mean(total_loss)
        msg = 'Validation loss for Epoch %d is %f' %(epoch,total_loss)
        print msg
        log('validation_loss_v2.log',msg)


    #main loop for training:
    print 'Start training model'
    for epoch in xrange(nb_epoch):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test(epoch)
        #save check point
        checkpoint = './checkpoints_v2/model_epoch_' + str(epoch) +'.pth'
        torch.save(model.state_dict(),checkpoint)

def evaluate():
    #parameters
    nb_epoch = 100
    img_rows,img_cols = 480, 640
    batch_size = 1
    imgDir = '/home/luoyan/project/dataset/mscoco/images/train2014'
    mapDir = '/home/chenshi/VQA/dataset/vqahat/vqahat_train'
    annFile = '/home/chenshi/VQA/dataset/VQA_1/annotation/mscoco_train2014_annotations.json'
    quesFile = '/home/chenshi/VQA/dataset/VQA_1/question/OpenEnded_mscoco_train2014_questions.json'
    embeddingDir = '/home/chenshi/VQA/dataset/word_embedding/GoogleNews-vectors-negative300.bin'

    if args.save_map and not os.path.exists(args.save_dir):
        print 'Saving data mode'
        os.mkdir(args.save_dir)

    #initialize data loader
    DataLoader = Batch_generator(img_rows,img_cols,batch_size,imgDir,mapDir,annFile,quesFile,embeddingDir,val_split=0.2)

    #defining model
    if args.model == 'normal':
        model = Model()
    elif args.model == 'attention':
        model = Model_Att()
    elif args.model == 'addition':
        model = Model_Add()
    else:
        assert 0, 'Invalid model selected'
    model.load_state_dict(torch.load(args.weights)) #loading pretrained weights
    model.cuda()
    model.eval()

    #metric initialzation
    metrics = ['cc','kld','sim','auc']
    score = init_metrics(metrics)
    #main loop for evluation
    print 'Start evaluating the results'
    for batch_idx in xrange(len(DataLoader.val_idx)):
        img, que, target,flag = DataLoader.get_batch(batch_idx,mode='val')
        if flag == 0:
            continue
        img, que, target = torch.from_numpy(img), torch.from_numpy(que), torch.from_numpy(target)
        img, que, target = Variable(img,volatile=True), Variable(que,volatile=True), Variable(target,volatile=True)
        img, que, target = img.cuda(), que.cuda(), target.cuda()
        output = model(img,que)

        output,target = process_map(output,target,img_rows,img_cols,args.save_map,args.save_dir,batch_idx)
        score['cc'] += cal_cc_score(output,target)
        score['kld'] += cal_kld_score(output,target)
        score['sim'] += cal_sim_score(output,target)
        # score['spearman'] += cal_spearman_corr(output,target)
        score['auc'] += cal_auc(output,target)
        score['count'] += 1

    for metric_ in score:
        if not metric_ == 'count':
            score[metric_] /= score['count']

    print 'AUC score: %f' %score['auc']
    print 'CC score: %f' %score['cc']
    print 'SIM score: %f' %score['sim']
    print 'KLD score: %f' %score['kld']
    # print 'Spearman Rank Correlation score: %f' %score['spearman']

if args.mode == 'training':
    train()
elif args.mode == 'eval':
    evaluate()
else:
    print 'Invalid mode'
