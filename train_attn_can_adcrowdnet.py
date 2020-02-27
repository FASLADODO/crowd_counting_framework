from args_util import context_aware_network_args_parse
from data_flow import get_train_val_list, get_dataloader, create_training_image_list
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, MeanAbsoluteError, MeanSquaredError
from ignite.engine import Engine
from ignite.handlers import Checkpoint, DiskSaver
from crowd_counting_error_metrics import CrowdCountingMeanAbsoluteError, CrowdCountingMeanSquaredError
from visualize_util import get_readable_time

import torch
from torch import nn
from models import AttnCanAdcrowdNet
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
    test_list = None

    # create data loader
    train_loader, val_loader, test_loader = get_dataloader(train_list, val_list, test_list, dataset_name="shanghaitech_keepfull")


    # model
    model = AttnCanAdcrowdNet()
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
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        timestamp = get_readable_time()
        print(timestamp + " Validation set Results - Epoch: {}  Avg mae: {:.2f} Avg mse: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['mae'], metrics['mse'], metrics['nll']))

    def score_function(engine):
        """
        saver score function
        :param engine:
        :return:
        """
        engine.state.metrics['mae']

    # docs on save and load
    to_save = {'trainer': trainer, 'model': model, 'optimizer': optimizer}
    save_handler = Checkpoint(to_save, DiskSaver('saved_model/' + args.task_id, create_dir=True, atomic=True),
                              filename_prefix=args.task_id,
                              n_saved=5,
                              score_function=score_function,
                              score_name="mae")

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=3), save_handler)

    trainer.run(train_loader, max_epochs=args.epochs)