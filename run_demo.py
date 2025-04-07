# By Yuxiang Sun, Dec. 14, 2020
# Email: sun.yuxiang@outlook.com

import os, argparse, time, datetime, sys, shutil, stat, torch
import numpy as np 
from torch.autograd import Variable
from torch.utils.data import DataLoader
from util.MF_dataset import MF_dataset 
from util.util import compute_results, visualize
from sklearn.metrics import confusion_matrix
from scipy.io import savemat 
from model import RTFNet
import torch.nn
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Test with pytorch')

parser.add_argument('--model_name', '-m', type=str, default='RTFNet')
parser.add_argument('--weight_name', '-w', type=str, default='RTFNET_152') # RTFNet_152, RTFNet_50, please change the number of layers in the network file
parser.add_argument('--file_name', '-f', type=str, default='final.pth')
parser.add_argument('--dataset_split', '-d', type=str, default='test') # test, test_day, test_night
parser.add_argument('--img_height', '-ih', type=int, default=480)
parser.add_argument('--img_width', '-iw', type=int, default=640)  
parser.add_argument('--num_workers', '-j', type=int, default=8)
parser.add_argument('--n_class', '-nc', type=int, default=9)
parser.add_argument('--data_dir', '-dr', type=str, default='./dataset/')
parser.add_argument('--model_dir', '-wd', type=str, default='./weights_backup/')
args = parser.parse_args()

 
if __name__ == '__main__':
  
    args.gpu=-1
    os.chmod("./runs", stat.S_IRWXO)
    model_dir = os.path.join(args.model_dir, args.weight_name)
    # if os.path.exists(model_dir) is False:
    #     sys.exit("the %s does not exit." %(model_dir))
    model_file = os.path.join(model_dir, args.file_name)


    conf_total = np.zeros((args.n_class, args.n_class))
    model = eval(args.model_name)(n_class=args.n_class)
    model = model.to('cpu')

    pretrained_weight = torch.load('/Users/qiaopeng/Desktop/多源信号/多模态/RTFNet-master/weights_backup/RTFNet_152/final.pth', map_location='cpu', weights_only=True)
    own_state = model.state_dict()
    for name, param in pretrained_weight.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)

    batch_size = 1
    test_dataset  = MF_dataset(data_dir=args.data_dir, split=args.dataset_split, input_h=args.img_height, input_w=args.img_width)
    test_loader  = DataLoader(
        dataset     = test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    ave_time_cost = 0.0

    model.eval()
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images).to('cpu')
            labels = Variable(labels).to('cpu')
            start_time = time.time()
            logits = model(images)  # logits.size(): mini_batch*num_class*480*640
            end_time = time.time()
            if it>=5: # # ignore the first 5 frames
                ave_time_cost += (end_time-start_time)
            # convert tensor to numpy 1d array
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(1).cpu().numpy().squeeze().flatten() # prediction and label are both 1-d array, size: minibatch*640*480
            # generate confusion matrix frame-by-frame
            # conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0,1,2,3,4,5,6,7,8]) # conf is an n_class*n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            # conf_total += conf
            # save demo images
            visualize(image_name=names, predictions=logits.argmax(1), weight_name=args.weight_name)
            print("frame %d/%d, %s, time cost: %.2f ms, demo result saved."
                  %(it+1, len(test_loader), names, (end_time-start_time)*1000))


