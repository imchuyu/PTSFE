import torch
from PTSFE import PTSFE
from torch import nn
from torch.utils.data import DataLoader
from utils import SetSeed,train,dataset

seeds=[2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71]
device = torch.device('cuda:7' if torch.cuda.is_available() else "cpu")
traindir = 'data/train'
valdir = 'data/val'
minibatch_size = 4
for seed in seeds:
    SetSeed(seed)
    trainnet = PTSFE(version='S')
    loss_fn = nn.CrossEntropyLoss()
    op = torch.optim.Adam(trainnet.parameters())
    traindata = dataset(traindir)
    valdata = dataset(valdir)
    trainloader = DataLoader(traindata, minibatch_size, shuffle=True, num_workers=6)
    valloader = DataLoader(valdata, batch_size=1)
    trainnet.to(device)
    trainnet = nn.DataParallel(trainnet, device_ids=[4, 5, 6, 7])
    loss_fn.to(device)
    test_data_size = len(valdata)
    train(seed, trainnet, trainloader, valloader, device, loss_fn, op, test_data_size)

