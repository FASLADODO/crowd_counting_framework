from comet_ml import Experiment

from args_util import my_args_parse
from data_flow import get_train_val_list, get_dataloader, create_training_image_list, create_image_list
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, MeanAbsoluteError, MeanSquaredError
from ignite.engine import Engine
from ignite.handlers import Checkpoint, DiskSaver
from crowd_counting_error_metrics import CrowdCountingMeanAbsoluteError, CrowdCountingMeanSquaredError
from visualize_util import get_readable_time

import torch
from torch import nn
from models import AttnCanAdcrowdNetSimpleV5
import os

from ignite.contrib.handlers import PiecewiseLinear
from model_util import get_lr
from torchsummary import summary

COMET_ML_API = "S3mM1eMq6NumMxk2QJAXASkUM"
PROJECT_NAME = "crowd-counting-framework"

if __name__ == "__main__":
    experiment = Experiment(project_name=PROJECT_NAME, api_key=COMET_ML_API)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args = my_args_parse()
    print(args)

    experiment.set_name(args.task_id)
    experiment.set_cmd_args()

    DATA_PATH = args.input
    TRAIN_PATH = os.path.join(DATA_PATH, "train_data")
    TEST_PATH = os.path.join(DATA_PATH, "test_data")
    dataset_name = args.datasetname
    if dataset_name=="shanghaitech":
        print("will use shanghaitech dataset with crop ")
    elif dataset_name == "shanghaitech_keepfull":
        print("will use shanghaitech_keepfull")
    else:
        print("cannot detect dataset_name")
        print("current dataset_name is ", dataset_name)

    # create list
    train_list = create_image_list(TRAIN_PATH)
    test_list = create_image_list(TEST_PATH)

    # create data loader
    train_loader, val_loader, test_loader = get_dataloader(train_list, None, test_list, dataset_name=dataset_name)

    print("len train_loader ", len(train_loader))

    # model
    model = AttnCanAdcrowdNetSimpleV5()
    experiment.log_other("model_summary", summary(model, (3, 128, 128), device="cpu"))
    model = model.to(device)

    # loss function
    loss_fn = nn.MSELoss(reduction='sum').to(device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.decay)

    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={
                                                'mae': CrowdCountingMeanAbsoluteError(),
                                                'mse': CrowdCountingMeanSquaredError(),
                                                'loss': Loss(loss_fn)
                                            }, device=device)
    print(model)

    print(args)

    milestones_values = [(10, 1e-4), (20, 1e-5), (60, 1e-5), (100, 1e-6)]
    experiment.log_parameter("milestones_values", str(milestones_values))
    lr_scheduler = PiecewiseLinear(optimizer, param_name="lr", milestones_values=milestones_values)

    if len(args.load_model) > 0:
        load_model_path = args.load_model
        print("load mode " + load_model_path)
        to_load = {'trainer': trainer, 'model': model, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        checkpoint = torch.load(load_model_path)
        Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)
        print("load model complete")
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
            print("change lr to ", args.lr)
    else:
        print("do not load, keep training")

    trainer.add_event_handler(Events.EPOCH_STARTED, lr_scheduler)


    @trainer.on(Events.ITERATION_COMPLETED(every=50))
    def log_training_loss(trainer):
        timestamp = get_readable_time()
        print(timestamp + " Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        timestamp = get_readable_time()
        print(timestamp + " Training set Results - Epoch: {}  Avg mae: {:.2f} Avg mse: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['mae'], metrics['mse'], metrics['loss']))
        experiment.log_metric("epoch", trainer.state.epoch)
        experiment.log_metric("train_mae", metrics['mae'])
        experiment.log_metric("train_mse", metrics['mse'])
        experiment.log_metric("train_loss", metrics['loss'])
        experiment.log_metric("lr", get_lr(optimizer))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        timestamp = get_readable_time()
        print(timestamp + " Validation set Results - Epoch: {}  Avg mae: {:.2f} Avg mse: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['mae'], metrics['mse'], metrics['loss']))
        experiment.log_metric("valid_mae", metrics['mae'])
        experiment.log_metric("valid_mse", metrics['mse'])
        experiment.log_metric("valid_loss", metrics['loss'])



    # docs on save and load
    to_save = {'trainer': trainer, 'model': model, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    save_handler = Checkpoint(to_save, DiskSaver('saved_model/' + args.task_id, create_dir=True, atomic=True),
                              filename_prefix=args.task_id,
                              n_saved=5)

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=3), save_handler)

    trainer.run(train_loader, max_epochs=args.epochs)
