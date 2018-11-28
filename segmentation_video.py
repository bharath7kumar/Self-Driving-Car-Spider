import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from skimage import segmentation
import torch.nn.init

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--nChannel', metavar='N', default=100, type=int, 
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=100, type=int, 
                    help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=7, type=int, 
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float, 
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int, 
                    help='number of convolutional layers')
parser.add_argument('--num_superpixels', metavar='K', default=1000, type=int, 
                    help='number of superpixels')
parser.add_argument('--compactness', metavar='C', default=100, type=float, 
                    help='compactness of superpixels')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, 
                    help='visualization flag')
parser.add_argument('--input', metavar='FILENAME',
                    help='input video file name', required=True)
args = parser.parse_args()

# CNN model
class MyNet(nn.Module):
    def __init__(self,input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = []
        self.bn2 = []
        for i in range(args.nConv-1):
            self.conv2.append( nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(args.nChannel) )
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(args.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

# load video
cap=cv2.VideoCapture(0)
while(True):

    ret,im = cap.read()
    cv2.imshow("original image",im)
    data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]) )
    if use_cuda:
        data = data.cuda()
    data = Variable(data)

    # slic
    labels = segmentation.slic(im, compactness=args.compactness, n_segments=args.num_superpixels)
    labels = labels.reshape(im.shape[0]*im.shape[1])
    u_labels = np.unique(labels)
    l_inds = []
    for i in range(len(u_labels)):
        l_inds.append( np.where( labels == u_labels[ i ] )[ 0 ] )

    # train
    model=torch.load('/home/hariharan/Desktop/Spider/Club Project/Self Driving  car/model.pt')
    label_colours = np.random.randint(255,size=(100,3))

    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
    im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
    
    cv2.imshow( "output", im_target_rgb )

    if cv2.waitKey(1) & 0XFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()