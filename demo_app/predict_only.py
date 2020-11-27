import os
from data_flow import get_predict_dataloader
if __name__ == "__main__":
    """
    predict all in folder 
    output into another folder 
    output density map and count in csv
    """
    INPUT_FOLDER = "/data/ShanghaiTech/part_B/test_data/images/"
    OUTPUT_FOLDER = "/data/apps/tmp"
    input_list = [os.path.join(INPUT_FOLDER, dir) for dir in os.listdir(INPUT_FOLDER)]
    loader = get_predict_dataloader(input_list)
    for img, info in loader:
        print(img.shape)
        print(info)