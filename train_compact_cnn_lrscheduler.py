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
from models import CompactCNN
import os
from ignite.contrib.handlers.param_scheduler import LRScheduler
from torch.optim.lr_scheduler import StepLR

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args = my_args_parse()
    print(args)
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
    model = CompactCNN()
    model = model.to(device)

    # loss function
    loss_fn = nn.MSELoss(reduction='sum').to(device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.decay)

    step_scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    lr_scheduler = LRScheduler(step_scheduler)

    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={
                                                'mae': CrowdCountingMeanAbsoluteError(),
                                                'mse': CrowdCountingMeanSquaredError(),
                                                'nll': Loss(loss_fn)
                                            }, device=device)
    print(model)

    print(args)

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
        trainer.add_event_handler(Events.ITERATION_COMPLETED, lr_scheduler)


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
              .format(trainer.state.epoch, metrics['mae'], metrics['mse'], metrics['nll']))


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        timestamp = get_readable_time()
        print(timestamp + " Validation set Results - Epoch: {}  Avg mae: {:.2f} Avg mse: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['mae'], metrics['mse'], metrics['nll']))



    # docs on save and load
    to_save = {'trainer': trainer, 'model': model, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    save_handler = Checkpoint(to_save, DiskSaver('saved_model/' + args.task_id, create_dir=True, atomic=True),
                              filename_prefix=args.task_id,
                              n_saved=5)

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=3), save_handler)

    trainer.run(train_loader, max_epochs=args.epochs)
