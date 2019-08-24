from args_util import real_args_parse
from data_flow import get_train_val_list, get_dataloader
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import torch
from torch import nn
from models import CSRNet

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args = real_args_parse()
    DATA_PATH = args.input

    # create list
    train_list, val_list = get_train_val_list(DATA_PATH)
    # create data loader
    train_loader, val_loader = get_dataloader(train_list, val_list)

    # model
    model = CSRNet()
    model = model.to(device)

    # loss function
    loss_fn = nn.MSELoss(size_average=False).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)

    trainer = create_supervised_trainer(model, optimizer, loss_fn)

    print(model)

    print(args)


    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))

    trainer.run(train_loader, max_epochs=1)