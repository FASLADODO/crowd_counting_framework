from comet_ml import Experiment
from torchvision.io.video import read_video, write_video
import torch
from args_util import meow_parse
from data_flow import get_predict_video_dataloader
from models import create_model
import os
from visualize_util import save_density_map_normalize, save_density_map


if __name__ == "__main__":

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
    # args = meow_parse()
    # print(args)
    # input_path = args.input
    input_path = video_path
    loader = get_predict_video_dataloader(input_path, batch_size=args.batch_size)
    single_frame = None
    model = create_model(model_name)
    loaded_file = torch.load(MODEL_PATH)
    model.load_state_dict(loaded_file['model'])
    model.to(device)
    os.makedirs(os.path.join(OUTPUT_FOLDER, NAME), exist_ok=True)
    log_file = open(os.path.join(OUTPUT_FOLDER, NAME, NAME + ".log"), 'w')
    count = 0
    experiment.log_other("total length", len(loader))
    for frame, info in loader:
        frame = frame.to(device)
        experiment.log_metric("count", count)
        # print("meow")
        pred = model(frame)
        index = count
        if args.batch_size == 1:
            index = str(info['index'][0].item())
        predict_name = "PRED_" + str(index)
        predict_path = os.path.join(OUTPUT_FOLDER, NAME, predict_name)
        pred = model(frame)
        if args.batch_size == 1:
            pred_np = pred.detach().cpu().numpy()[0][0]
            pred_count = pred_np.sum()
            log_line = str(index) + "," + str(pred_count.item()) +"\n"
            log_file.write(log_line)
            save_density_map(pred_np, predict_path)
        torch.save(pred, predict_path+".torch")
        print("save to ", predict_path)
        count += 1
    log_file.close()

    print(single_frame)
