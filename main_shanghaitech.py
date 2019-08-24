from args_util import real_args_parse
from data_flow import get_train_val_list, get_dataloader, create_training_image_list
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, MeanAbsoluteError, MeanSquaredError
from crowd_counting_error_metrics import CrowdCountingMeanAbsoluteError, CrowdCountingMeanSquaredError
import torch
from torch import nn
from models import CSRNet
import os

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args = real_args_parse()
    print(args)
    DATA_PATH = args.input
    TRAIN_PATH = os.path.join(DATA_PATH, "train_data")
    TEST_PATH = os.path.join(DATA_PATH, "test_data")


    # create list
    train_list, val_list = get_train_val_list(TRAIN_PATH)
    test_list = create_training_image_list(TEST_PATH)

    # create data loader
    train_loader, val_loader, test_loader = get_dataloader(train_list, val_list, test_list)


    # model
    model = CSRNet()
    model = model.to(device)

    # loss function
    loss_fn = nn.MSELoss(size_average=False).cuda()

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


    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {}  Avg mae: {:.2f} Avg mse: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['mae'], metrics['mse'], metrics['nll']))


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {}  Avg mae: {:.2f} Avg mse: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['mae'], metrics['mse'], metrics['nll']))


    trainer.run(train_loader, max_epochs=10)