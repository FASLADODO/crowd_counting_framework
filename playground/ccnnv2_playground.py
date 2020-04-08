from models import CompactCNNV2
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
    dataset_name = "shanghaitech_256"

    count_below_256 = 0
    # create list
    train_list, val_list = get_train_val_list(TRAIN_PATH)
    test_list = None

    # create data loader
    train_loader, val_loader, test_loader = get_dataloader(train_list, val_list, test_list, dataset_name=dataset_name, batch_size=5)
    model = CompactCNNV2()
    # model = model.cuda()
    print("============== TRAIN LOADER ====================================================")
    min_1 = 500
    min_2 = 500
    for img, label in train_loader:
        out = model(img)
        print(out.shape)
        exit()