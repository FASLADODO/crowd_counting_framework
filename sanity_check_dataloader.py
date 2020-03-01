from args_util import sanity_check_dataloader_parse
from data_flow import get_train_val_list, get_dataloader, create_training_image_list
import torch
import os


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args = sanity_check_dataloader_parse()
    print(args)
    DATA_PATH = args.input
    TRAIN_PATH = os.path.join(DATA_PATH, "train_data")
    TEST_PATH = os.path.join(DATA_PATH, "test_data")
    dataset_name = args.datasetname


    # create list
    train_list, val_list = get_train_val_list(TRAIN_PATH)
    test_list = None

    # create data loader
    train_loader, val_loader, test_loader = get_dataloader(train_list, val_list, test_list, dataset_name=dataset_name)

    print("============== TRAIN LOADER ====================================================")
    for img, label in train_loader:
        print("img shape:" + str(img.shape) + " == " + "label shape " +  str(label.shape))
    print("============== VAL LOADER ====================================================")
    for img, label in val_loader:
        print("img shape:" + str(img.shape) + " == " + "label shape " +  str(label.shape))
