import os
import torch
from data_flow import get_predict_dataloader
from models.dccnn import DCCNN
from models.compact_cnn import CompactCNNV7
from visualize_util import save_density_map_normalize, save_density_map
from comet_ml import Experiment
from args_util import meow_parse

if __name__ == "__main__":
    """
    predict all in folder 
    output into another folder 
    output density map and count in csv
    """

    COMET_ML_API = "S3mM1eMq6NumMxk2QJAXASkUM"
    PROJECT_NAME = "crowd-counting-debug"

    experiment = Experiment(project_name=PROJECT_NAME, api_key=COMET_ML_API)

    args = meow_parse()
    video_path = args.input
    OUTPUT_FOLDER = args.output
    MODEL_PATH = args.load_model
    model_name = args.model
    NAME = args.task_id

    experiment.set_name(args.task_id)
    experiment.set_cmd_args()
    experiment.log_text(args.note)

    print(args)
    n_thread = int(os.environ['OMP_NUM_THREADS'])
    torch.set_num_threads(n_thread)  # 4 thread
    print("n_thread ", n_thread)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    NAME = args.task_id
    INPUT_FOLDER = args.input
    OUTPUT_FOLDER = args.output
    MODEL = args.model
    input_list = [os.path.join(INPUT_FOLDER, dir) for dir in os.listdir(INPUT_FOLDER)]
    loader = get_predict_dataloader(input_list)
    loaded_file = torch.load(MODEL)
    model = CompactCNNV7()
    model.load_state_dict(loaded_file['model'])
    model.eval()
    model = model.to(device)
    os.makedirs(os.path.join(OUTPUT_FOLDER, NAME), exist_ok=True)
    log_file = open(os.path.join(OUTPUT_FOLDER, NAME, NAME +".log"), 'w')
    # limit_count = 100
    count = 0
    for img, info in loader:
        # if count > limit_count:
        #     break
        predict_name = "PRED_" + info["name"][0]
        img = img.to(device)
        predict_path = os.path.join(OUTPUT_FOLDER, NAME, predict_name)
        pred = model(img)
        pred = pred.detach().cpu().numpy()[0][0]
        pred_count = pred.sum()
        log_line = info["name"][0] + "," + str(pred_count.item()) +"\n"
        log_file.write(log_line)
        save_density_map(pred, predict_path)
        torch.save(pred, predict_path+".torch")
        print("save to ", predict_path)
        print(log_line)
        count += 1
    log_file.close()
