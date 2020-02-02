import h5py
import PIL.Image as Image
import numpy as np
import os
import glob
import torch
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error,mean_absolute_error
from torchvision import transforms
from models.context_aware_network import CANNet
from data_util import ShanghaiTechDataPath
from hard_code_variable import HardCodeVariable

_description="""
This file run predict
Data path = /home/tt/project/ShanghaiTechCAN/part_B/test_data/images
model path = /home/tt/project/MODEL/Context-aware/part_B_pre.pth.tar
"""

transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

# the folder contains all the test images
hard_code = HardCodeVariable()
shanghaitech_data = ShanghaiTechDataPath(root=hard_code.SHANGHAITECH_PATH)
# img_folder='/home/tt/project/ShanghaiTechCAN/part_B/test_data/images'
img_folder = shanghaitech_data.get_b().get_test().get_images()
print("image folder = " + str(img_folder))

img_paths=[]

for img_path in glob.glob(os.path.join(img_folder, '*.jpg')):
    img_paths.append(img_path)
# img_paths = img_paths[:10]


model = CANNet()

model = model.cuda()

checkpoint = torch.load('/home/tt/project/MODEL/Context-aware/part_B_pre.pth.tar')

model.load_state_dict(checkpoint['state_dict'])

model.eval()

pred= []
gt = []

for i in range(len(img_paths)):
    img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
    img = img.unsqueeze(0)
    h,w = img.shape[2:4]
    h_d = int(h/2)
    w_d = int(w/2)
    img_1 = Variable(img[:,:,:h_d,:w_d].cuda())
    img_2 = Variable(img[:,:,:h_d,w_d:].cuda())
    img_3 = Variable(img[:,:,h_d:,:w_d].cuda())
    img_4 = Variable(img[:,:,h_d:,w_d:].cuda())
    density_1 = model(img_1).data.cpu().numpy()
    density_2 = model(img_2).data.cpu().numpy()
    density_3 = model(img_3).data.cpu().numpy()
    density_4 = model(img_4).data.cpu().numpy()

    pure_name = os.path.splitext(os.path.basename(img_paths[i]))[0]
    gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground-truth-h5'),'r')
    groundtruth = np.asarray(gt_file['density'])
    pred_sum = density_1.sum()+density_2.sum()+density_3.sum()+density_4.sum()
    pred.append(pred_sum)
    gt.append(np.sum(groundtruth))
    print("done ", i, "pred ",pred_sum, " gt ", np.sum(groundtruth))

print(len(pred))
print(len(gt))
mae = mean_absolute_error(pred,gt)
rmse = np.sqrt(mean_squared_error(pred,gt))

print('MAE: ',mae)
print('RMSE: ',rmse)
