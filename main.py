from args_util import real_args_parse
from data_flow import get_train_val_list, get_dataloader
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
if __name__ == "__main__":
    args = real_args_parse()
    DATA_PATH = args.input

    # create list
    train_list, val_list = get_train_val_list(DATA_PATH)
    # create data loader
    train_loader, test_loader = get_dataloader(train_list, val_list)



    print(args)