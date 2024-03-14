import os
import torch
import nibabel as nib
import numpy as np
import random
import sys
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Dataset


class dataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.imagelist = os.listdir(path)

    def __getitem__(self, item):
        image_name = self.imagelist[item]
        lable = self.imagelist[item][0]
        if lable == "i":
            lable = torch.Tensor([1, 0])
        else:
            lable = torch.Tensor([0, 1])
        imagepath = os.path.join(self.path, image_name)
        image = nib.load(imagepath)
        image = nii2tensor(image)
        return image, lable

    def __len__(self):
        return len(self.imagelist)


def resampling(data):
    size_X = 256
    size_Y = 256
    size_Z = 128
    data = torch.tensor(data).unsqueeze(0).unsqueeze(0)
    data = data.to(torch.float32)
    data = torch.nn.functional.interpolate(data,
                                           size=[size_X, size_Y, size_Z],
                                           mode='trilinear')
    return data[0, 0, :, :, :]


def nii2tensor(img):
    img_data = np.asarray(img.dataobj)
    img_hdr = img.header
    voxel = [img_hdr['pixdim'][1]]
    voxel.append(img_hdr['pixdim'][2])
    voxel.append(img_hdr['pixdim'][3])
    resample_spacing = [1, 1, 1]
    img_isotropic = resampling(img_data)
    scaling_affine = np.diag([resample_spacing[0], resample_spacing[1], resample_spacing[2], 1])
    img_new = nib.Nifti1Image(img_isotropic, scaling_affine, img_hdr)
    img_data = np.asarray(img_new.dataobj)
    return  torch.from_numpy(img_data).unsqueeze(0)


def SetSeed(seed): #随机数种子1029
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def train(seed,trainnet,trainloader,valloader,device,loss_fn,op,test_data_size):
    best = 0.0
    epoch=100
    for i in range(epoch):
        trainnet.train()
        running_loss = 0.0
        train_bar = tqdm(trainloader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            op.zero_grad()
            outputs = trainnet(images.to(device))
            loss = loss_fn(outputs, labels.to(device))
            loss.backward()
            op.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(i + 1,epoch,loss)
        trainnet.eval()
        total_accuracy = 0.0
        with torch.no_grad():
            val_bar = tqdm(valloader, file=sys.stdout)
            y_true = []
            y_scores = []
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_labels=val_labels.to(device)
                outputs = trainnet(val_images.to(device))
                accuracy = (outputs.argmax(1) == val_labels.argmax(1)).sum()
                y_scores.append(torch.softmax(outputs[0], 0)[0].to("cpu"))
                y_true.append(val_labels[0][0].to("cpu"))
                total_accuracy = total_accuracy + accuracy
        val_accurate = total_accuracy / test_data_size
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        score=(val_accurate+roc_auc)/2
        if score >=best:
            best = score
            torch.save(trainnet, 'seed'+str(seed)+'.pth')
        print('[epoch %d] train_loss: %.3f  val_score: %.3f  best: %.3f' %
              (i + 1, running_loss, score, best))
        if running_loss<=0.01:
            break
    print("seed"+seed+'Finished Training')