import torch
from utils import dataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc


device = torch.device('cuda:7' if torch.cuda.is_available() else "cpu")
trainnet=torch.load("PTSFE_B.pth")
trainnet.to(device)
testdata=dataset("data/test")
testloader=DataLoader(testdata,batch_size=1)
test_data_size = len(testdata)
y_true =[]
y_scores = []
trainnet.eval()
total_accuracy = 0
total_tp=0
total_fn=0
total_fp=0
with torch.no_grad():
    for data in testloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = trainnet(imgs)
        accuracy = (outputs.argmax(1) == targets.argmax(1)).sum()
        tp=(outputs.argmax(1) == 1 and targets.argmax(1)==1).sum()
        fp=(outputs.argmax(1) == 1 and targets.argmax(1)==0).sum()
        fn=(outputs.argmax(1) == 0 and targets.argmax(1)==1).sum()
        y_scores.append(torch.softmax(outputs[0],0)[0].to("cpu"))
        y_true.append(targets[0][0].to("cpu"))
        total_accuracy = total_accuracy + accuracy
        total_fp+=fp
        total_fn+=fn
        total_tp+=tp
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
acc=total_accuracy / test_data_size
f1=2*total_tp/(2*total_tp+total_fn+total_fp)
print(roc_auc,acc,f1)