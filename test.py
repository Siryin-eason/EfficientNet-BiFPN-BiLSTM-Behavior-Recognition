"""
author: Sir-yin-einson
motto: No pain, no gain
"""
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from myfunctions import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
# set path
data_path = "xxx"             # Fllow UCF-101 RGB data path
action_name_path = "./label.pkl"
save_model_path = "./weights"
# use same encoder CNN saved!
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
res_size = 512       # ResNet image size
dropout_p = 0.0       # dropout probability

# use same decoder RNN saved!
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# training parameters
k =5         # number of target category
batch_size = 1
# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 64, 4


with open(action_name_path, 'rb') as f:
    action_names = pickle.load(f)   # load UCF101 actions names

# convert labels -> category
le = LabelEncoder()
le.fit(action_names)

# show how many classes there are
list(le.classes_)

# convert category -> 1-hot
action_category = le.transform(action_names).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(action_category)

actions = []
fnames = os.listdir(data_path)

all_names = []
for f in fnames:
    loc2 = f.find('_')
    actions.append(f[: loc2])

    all_names.append(f)


# list all data files
all_X_list = all_names              # all video file names
all_y_list = labels2cat(le, actions)    # all video labels

# data loading parameters
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
torch.cuda.set_device(0)
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0, 'pin_memory': True} if use_cuda else {}


transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

# reset data loader
all_data_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 0, 'pin_memory': True} if use_cuda else {}
all_data_loader = data.DataLoader(Dataset_CRNN(data_path, all_X_list, all_y_list, selected_frames, transform=transform), **all_data_params)


# reload CRNN model
cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)
##########################################修改为自己训练好的模型#########################
cnn_encoder.load_state_dict(torch.load(os.path.join(save_model_path, 'cnn_encoder_epoch1.pth')))
rnn_decoder.load_state_dict(torch.load(os.path.join(save_model_path, 'rnn_decoder_epoch1.pth')))
print('CRNN model reloaded!')


# make all video predictions by reloaded model
print('Predicting all {} videos:'.format(len(all_data_loader.dataset)))
all_y_pred = CRNN_final_prediction([cnn_encoder, rnn_decoder], device, all_data_loader)


# write in pandas dataframe
df = pd.DataFrame(data={'filename': fnames, 'y': cat2labels(le, all_y_list), 'y_pred': cat2labels(le, all_y_pred)})
df.to_pickle("./my_videos_prediction.pkl")  # save pandas dataframe
# pd.read_pickle("./all_videos_prediction.pkl")
print('video prediction finished!')




