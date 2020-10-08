import argparse
import logging
import os

import numpy as np 
import torch 
import torch.nn.functional as F
from PIL import Image
import cv2
from torchvision import transforms

from model import ResUNetA
from utils import plot_img_and_mask, preprocess
from dataset import Dataset
import matplotlib.pyplot as plt

class ToNumpy(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = value.transpose((2, 0, 1)).numpy()
        #
        # return data
        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

        # input, label = data['input'], data['label']
        # input = input.transpose((2, 0, 1))
        # label = label.transpose((2, 0, 1))
        # return {'input': input.detach().numpy(), 'label': label.detach().numpy()}

class Denormalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data = self.std * data + self.mean
        return data
    
def predict_mask(net, input_img, device):
    
    # transform_inv = transforms.Compose([ToNumpy(), Denormalize()])
    transform_ts2np = ToNumpy()

    # img = torch.from_numpy(preprocess(input_img))
    img = transforms.ToTensor()(input_img).unsqueeze(0)
   
    # img = torch.from_numpy(img).to(device)
    # img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
      net.eval()
      output = net(img)
      output = transform_ts2np(torch.sigmoid(output))
      #  output = 1.0 * (output > 0.5)   # theshold 

    return output[0].squeeze()
    


def get_args():
    parser = argparse.ArgumentParser(description="Options parser for predict output", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model','-m', default="model.pth", metavar='FILE',)
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=True)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=True)
    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = args.output
     
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_img = Image.open(in_files[0])
    # print(input_img)
  
    net = ResUNetA(3,1)
    net.to(device=device)
    

    # print(args.model)
    load = torch.load(args.model, map_location=device)
    # print(load.keys())

    

    net.load_state_dict(load['netG'])
    
    mask = predict_mask(net, input_img, device)
    plot_img_and_mask(input_img, mask)

       



