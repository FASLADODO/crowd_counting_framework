import torch
from models.meow_experiment.ccnn_tail import BigTail11i, BigTail10i, BigTail12i, BigTail13i, BigTail14i, BigTail15i
from hard_code_variable import HardCodeVariable
from data_util import ShanghaiTechDataPath
from visualize_util import save_img, save_density_map
import os
import numpy as np
from data_flow import get_train_val_list, get_dataloader, create_training_image_list
import cv2

def visualize_evaluation_shanghaitech_keepfull(model):
    model = model.cuda()
    model.eval()
    HARD_CODE = HardCodeVariable()
    shanghaitech_data = ShanghaiTechDataPath(root=HARD_CODE.SHANGHAITECH_PATH)
    shanghaitech_data_part_a_train = shanghaitech_data.get_a().get_train().get()
    saved_folder = "visualize/evaluation_dataloader_shanghaitech"
    os.makedirs(saved_folder, exist_ok=True)
    train_list, val_list = get_train_val_list(shanghaitech_data_part_a_train, test_size=0.2)
    test_list = None
    train_loader, val_loader, test_loader = get_dataloader(train_list, val_list, test_list, dataset_name="shanghaitech_keepfull", visualize_mode=False,
                                                           debug=True)

    # do with train loader
    train_loader_iter = iter(train_loader)
    for i in range(10):
        img, label, count = next(train_loader_iter)
        # save_img(img, os.path.join(saved_folder, "train_img_" + str(i) +".png"))
        save_path = os.path.join(saved_folder, "train_label_"  + str(i) +".png")
        save_pred_path = os.path.join(saved_folder, "train_pred_" + str(i) +".png")
        save_density_map(label.numpy()[0][0], save_path)
        pred = model(img.cuda())
        predicted_density_map = pred.detach().cpu().clone().numpy()
        predicted_density_map_enlarge = cv2.resize(np.squeeze(predicted_density_map[0][0]), (int(predicted_density_map.shape[3] * 8), int(predicted_density_map.shape[2] * 8)), interpolation=cv2.INTER_CUBIC) / 64
        save_density_map(predicted_density_map_enlarge, save_pred_path)
        print("pred " + save_pred_path + " value " + str(predicted_density_map.sum()))
        print("cont compare " + str(predicted_density_map.sum()) + " " + str(predicted_density_map_enlarge.sum()))
        print("shape compare " + str(predicted_density_map.shape) + " " + str(predicted_density_map_enlarge.shape))
"""
Document on save load model 
https://pytorch.org/tutorials/beginner/saving_loading_models.html
"""

model_path = "/data/save_model/adamw1_bigtail13i_t1_shb/adamw1_bigtail13i_t1_shb_checkpoint_valid_mae=-7.574910521507263.pth"
checkpoint = torch.load(model_path)

model = BigTail13i()
model.load_state_dict(checkpoint["model"])
print("done load")
visualize_evaluation_shanghaitech_keepfull(model)

