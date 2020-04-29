from comet_ml import Experiment

from args_util import meow_parse
from data_flow import get_dataloader, create_image_list
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss
from ignite.handlers import Checkpoint, DiskSaver, Timer
from crowd_counting_error_metrics import CrowdCountingMeanAbsoluteError, CrowdCountingMeanSquaredError
from visualize_util import get_readable_time

import torch
from torch import nn
from models.meow_experiment.kitten_meow_1 import M1, M2, M3, M4
from models.meow_experiment.ccnn_tail import BigTailM1, BigTailM2, BigTail3
from models.meow_experiment.ccnn_head import H1
from models import CustomCNNv2
import os
from model_util import get_lr

COMET_ML_API = "S3mM1eMq6NumMxk2QJAXASkUM"
PROJECT_NAME = "meow-one-experiment-insita"
# PROJECT_NAME = "crowd-counting-debug"


def very_simple_param_count(model):
    result = sum([p.numel() for p in model.parameters()])
    return result


if __name__ == "__main__":
    experiment = Experiment(project_name=PROJECT_NAME, api_key=COMET_ML_API)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args = meow_parse()
    print(args)

    experiment.set_name(args.task_id)
    experiment.set_cmd_args()
    experiment.log_text(args.note)

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
    train_loader, train_loader_for_eval, test_loader = get_dataloader(train_list, train_list, test_list, dataset_name=dataset_name, batch_size=args.batch_size)

    print("len train_loader ", len(train_loader))

    # model
    model_name = args.model
    experiment.log_other("model", model_name)
    if model_name == "M1":
        model = M1()
    elif model_name == "M2":
        model = M2()
    elif model_name == "M3":
        model = M3()
    elif model_name == "M4":
        model = M4()
    elif model_name == "CustomCNNv2":
        model = CustomCNNv2()
    elif model_name == "BigTailM1":
        model = BigTailM1()
    elif model_name == "BigTailM2":
        model = BigTailM2()
    elif model_name == "BigTail3":
        model = BigTail3()
    elif model_name == "H1":
        model = H1()
    else:
        print("error: you didn't pick a model")
        exit(-1)
    n_param = very_simple_param_count(model)
    experiment.log_other("n_param", n_param)
    if hasattr(model, 'model_note'):
        experiment.log_other("model_note", model.model_note)
    model = model.to(device)

    # loss function
    loss_fn = nn.MSELoss(reduction='sum').to(device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.decay)

    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator_train = create_supervised_evaluator(model,
                                            metrics={
                                                'mae': CrowdCountingMeanAbsoluteError(),
                                                'mse': CrowdCountingMeanSquaredError(),
                                                'loss': Loss(loss_fn)
                                            }, device=device)

    evaluator_validate = create_supervised_evaluator(model,
                                            metrics={
                                                'mae': CrowdCountingMeanAbsoluteError(),
                                                'mse': CrowdCountingMeanSquaredError(),
                                                'loss': Loss(loss_fn)
                                            }, device=device)
    print(model)

    print(args)


    # timer
    train_timer = Timer(average=True)  # time to train whole epoch
    batch_timer = Timer(average=True)  # every batch
    evaluate_timer = Timer(average=True)

    batch_timer.attach(trainer,
                        start =Events.EPOCH_STARTED,
                        resume =Events.ITERATION_STARTED,
                        pause =Events.ITERATION_COMPLETED,
                        step =Events.ITERATION_COMPLETED)

    train_timer.attach(trainer,
                        start =Events.EPOCH_STARTED,
                        resume =Events.EPOCH_STARTED,
                        pause =Events.EPOCH_COMPLETED,
                        step =Events.EPOCH_COMPLETED)

    if len(args.load_model) > 0:
        load_model_path = args.load_model
        print("load mode " + load_model_path)
        to_load = {'trainer': trainer, 'model': model, 'optimizer': optimizer}
        checkpoint = torch.load(load_model_path)
        Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)
        print("load model complete")
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
            print("change lr to ", args.lr)
    else:
        print("do not load, keep training")


    @trainer.on(Events.ITERATION_COMPLETED(every=100))
    def log_training_loss(trainer):
        timestamp = get_readable_time()
        print(timestamp + " Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator_train.run(train_loader_for_eval)
        metrics = evaluator_train.state.metrics
        timestamp = get_readable_time()
        print(timestamp + " Training set Results - Epoch: {}  Avg mae: {:.2f} Avg mse: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['mae'], metrics['mse'], metrics['loss']))
        experiment.log_metric("epoch", trainer.state.epoch)
        experiment.log_metric("train_mae", metrics['mae'])
        experiment.log_metric("train_mse", metrics['mse'])
        experiment.log_metric("train_loss", metrics['loss'])
        experiment.log_metric("lr", get_lr(optimizer))

        experiment.log_metric("batch_timer", batch_timer.value())
        experiment.log_metric("train_timer", train_timer.value())

        print("batch_timer ", batch_timer.value())
        print("train_timer ", train_timer.value())

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluate_timer.resume()
        evaluator_validate.run(test_loader)
        evaluate_timer.pause()
        evaluate_timer.step()

        metrics = evaluator_validate.state.metrics
        timestamp = get_readable_time()
        print(timestamp + " Validation set Results - Epoch: {}  Avg mae: {:.2f} Avg mse: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['mae'], metrics['mse'], metrics['loss']))
        experiment.log_metric("valid_mae", metrics['mae'])
        experiment.log_metric("valid_mse", metrics['mse'])
        experiment.log_metric("valid_loss", metrics['loss'])

        # timer
        experiment.log_metric("evaluate_timer", evaluate_timer.value())
        print("evaluate_timer ", evaluate_timer.value())

    def checkpoint_valid_mae_score_function(engine):
        score = engine.state.metrics['mae']
        return score


    # docs on save and load
    to_save = {'trainer': trainer, 'model': model, 'optimizer': optimizer}
    save_handler = Checkpoint(to_save, DiskSaver('saved_model/' + args.task_id, create_dir=True, atomic=True),
                              filename_prefix=args.task_id,
                              n_saved=5)

    save_handler_best = Checkpoint(to_save, DiskSaver('saved_model_best/' + args.task_id, create_dir=True, atomic=True),
                              filename_prefix=args.task_id, score_name="valid_mae", score_function=checkpoint_valid_mae_score_function,
                              n_saved=5)

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=5), save_handler)
    evaluator_validate.add_event_handler(Events.EPOCH_COMPLETED(every=1), save_handler_best)


    trainer.run(train_loader, max_epochs=args.epochs)
