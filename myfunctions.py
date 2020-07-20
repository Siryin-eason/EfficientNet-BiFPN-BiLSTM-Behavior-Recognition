"""
author: Sir-yin-einson
motto: No pain, no gain
"""
import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from modle.efficientnet import EfficientNet,extra_CNN
from modle.bifpn import BIFPN
from tqdm import tqdm
import cv2
import matplotlib.pylab as plt
## ------------------- label conversion tools ------------------ ##
def labels2cat(label_encoder, list):
    return label_encoder.transform(list)

def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()

def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()

def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()

## ---------------------- Dataloaders ---------------------- ##
# for CRNN
class Dataset_CRNN(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data_path, folders, labels, frames, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, '{:06d}.jpg'.format(i)))

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(self.data_path, folder, self.transform)  # (input) spatial images
        y = torch.LongTensor([self.labels[index]])  # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X, y

## ---------------------- end of Dataloaders ---------------------- ##

## -------------------- (reload) model prediction ---------------------- ##

def CRNN_final_prediction(model, device, loader):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    all_y_pred = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)):
            # distribute data to device
            X = X.to(device)
            output = rnn_decoder(cnn_encoder(X))
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            # print("y_pred",[y_pred.cpu().data.squeeze().numpy()])
            all_y_pred.extend([y_pred.cpu().data.squeeze().numpy()])

    return all_y_pred


## -------------------- end of model prediction ---------------------- ##

## ------------------------ CRNN module ---------------------- ##
# 2D CNN encoder using EfficientNet pretrained
class ResCNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        ###############支持其他网络的嵌套##############
        # resnet = models.wide_resnet101_2(pretrained=True)
        # resnet = models.resnet152(pretrained=True)
        # resnet = models.densenet169(pretrained=True)
        # modules = list(resnet.children())[:-1]  # delete the last fc layer.
        #############################################
        resnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.resnet = resnet
        ###############BiFPN模块##############
        self.neck = BIFPN(in_channels=resnet.get_list_features()[-5:],
                     out_channels=64,
                     stack=2,
                     num_outs=5)
        # self.fc1 = nn.Linear(resnet.classifier.in_features, fc_hidden1)
        #self.fc1 = nn.Linear(resnet._fc.in_features, fc_hidden1)
        self.fc1 = nn.Linear(81920, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            with torch.no_grad():
                output = self.resnet(x_3d[:, t, :, :, :])[:]  # ResNet
                #############feature map可视化#####################
                """
                for i in range(len(output)):
                    for j in range(len(output[i][0])):
                        img=+output[i][:,j,:,:].squeeze(0).cpu().numpy()
                        print(img.shape)
                        cv2.imshow("img",img)
                        cv2.imwrite("./output"+str(i)+"_"+str(j)+".jpg",img*255)
                """
                ###################################################
                output1 = self.resnet(x_3d[:, t, :, :, :])[-5:]  # ResNet
                output = self.neck(output1)
                #################特征融合模块 #####################
                extra_CNN1 = extra_CNN(1 / 4)
                extra_CNN2 = extra_CNN(1 / 2)
                extra_CNN3 = extra_CNN(1)
                extra_CNN4 = extra_CNN(2)
                extra_CNN5 = extra_CNN(4)
                x1 = extra_CNN1(output[0])
                x2 = extra_CNN2(output[1])
                x3 = extra_CNN3(output[2])
                x4 = extra_CNN4(output[3])
                x5 = extra_CNN5(output[4])
                x = torch.cat([x1, x2, x3, x4, x5], 2)
                ###################################################
                #print(x.shape)
                x = x.view(x.size(0), -1)  # flatten output of conv
            # ResNet CNN
            # FC layers
            x=self.fc1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=50):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes
        ###############BiLSTM模块################
        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.5# input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        ###############Attention###############
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.Tensor(512 * 2))
        self.tanh2 = nn.Tanh()

        self.fc1 = nn.Linear(self.h_RNN* 2, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """
        M = self.tanh1(RNN_out)  # [10, 16, 1024]
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [10, 16, 1024]
        out = RNN_out* alpha  # [10, 16, 1024]
        RNN_out= torch.sum(out, 1)  # [10, 1024]
        # FC layers
        #x = self.fc1(RNN_out[:, -1, :])  # choose RNN_out at the last time step
        x = self.fc1(RNN_out)  # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x

## ---------------------- end of CRNN module ---------------------- ##

