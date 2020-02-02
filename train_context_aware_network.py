from args_util import context_aware_network_args_parse
from data_flow import get_train_val_list, get_dataloader, create_training_image_list
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, MeanAbsoluteError, MeanSquaredError
from ignite.engine import Engine
from ignite.handlers import Checkpoint, DiskSaver
from crowd_counting_error_metrics import CrowdCountingMeanAbsoluteError, CrowdCountingMeanSquaredError

import torch
from torch import nn
from models import CANNet
import os


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args = context_aware_network_args_parse()
    print(args)
    DATA_PATH = args.input
    TRAIN_PATH = os.path.join(DATA_PATH, "train_data")
    TEST_PATH = os.path.join(DATA_PATH, "test_data")


    # create list
    train_list, val_list = get_train_val_list(TRAIN_PATH)
    test_list = create_training_image_list(TEST_PATH)

    # create data loader
    train_loader, val_loader, test_loader = get_dataloader(train_list, val_list, test_list, dataset_name="shanghaitech_keepfull")


    # model
    model = CANNet()
    model = model.to(device)

    # loss function
    loss_fn = nn.MSELoss(size_average=False).to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)

    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={
                                                'mae': CrowdCountingMeanAbsoluteError(),
                                                'mse': CrowdCountingMeanSquaredError(),
                                                'nll': Loss(loss_fn)
                                            }, device=device)
    print(model)

    print(args)


    @trainer.on(Events.ITERATION_COMPLETED(every=50))
    def log_training_loss(trainer):
        print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print("Training set Results - Epoch: {}  Avg mae: {:.2f} Avg mse: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['mae'], metrics['mse'], metrics['nll']))


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print("Validation set Results - Epoch: {}  Avg mae: {:.2f} Avg mse: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['mae'], metrics['mse'], metrics['nll']))

    # docs on save and load
    to_save = {'trainer': trainer, 'model': model, 'optimizer': optimizer}
    save_handler = Checkpoint(to_save, DiskSaver('saved_model/context_aware_network', create_dir=True), filename_prefix=args.task_id)
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), save_handler)

    trainer.run(train_loader, max_epochs=args.epochs)