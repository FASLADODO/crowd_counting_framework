from comet_ml import Experiment
import sys
from args_util import meow_parse, lr_scheduler_milestone_builder
from data_flow import get_dataloader, create_image_list
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss
from ignite.handlers import Checkpoint, DiskSaver, Timer
from crowd_counting_error_metrics import CrowdCountingMeanAbsoluteError, CrowdCountingMeanSquaredError,\
CrowdCountingMeanAbsoluteErrorWithCount, CrowdCountingMeanSquaredErrorWithCount,\
CrowdCountingMeanSSIMabs, CrowdCountingMeanPSNRabs, \
CrowdCountingMeanSSIMclamp, CrowdCountingMeanPSNRclamp

from visualize_util import get_readable_time
from mse_l1_loss import MSEL1Loss, MSE4L1Loss
import torch
from torch import nn
from models.meow_experiment.kitten_meow_1 import M1, M2, M3, M4
from models.csrnet import CSRNet
from models.meow_experiment.ccnn_tail import BigTailM1, BigTailM2, BigTail3, BigTail4, BigTail5, BigTail6, BigTail7, BigTail8, BigTail6i, BigTail9i
from models.meow_experiment.ccnn_tail import BigTail11i, BigTail10i, BigTail12i, BigTail13i, BigTail14i, BigTail15i
from models.meow_experiment.ccnn_head import H1, H2, H3, H3i, H4i
from models.meow_experiment.kitten_meow_1 import H1_Bigtail3
from models import CustomCNNv2, CompactCNNV7
from models.compact_cnn import CompactCNNV8, CompactCNNV9, CompactCNNV7i
import os
from model_util import get_lr, BestMetrics
from ignite.contrib.handlers import PiecewiseLinear

COMET_ML_API = "S3mM1eMq6NumMxk2QJAXASkUM"
PROJECT_NAME = "crowd-counting-train-val"
# PROJECT_NAME = "crowd-counting-debug"


def very_simple_param_count(model):
    result = sum([p.numel() for p in model.parameters()])
    return result


if __name__ == "__main__":

    n_thread = int(os.environ['OMP_NUM_THREADS'])
    torch.set_num_threads(n_thread)  # 4 thread
    print("n_thread ", n_thread)
    experiment = Experiment(project_name=PROJECT_NAME, api_key=COMET_ML_API)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args = meow_parse()
    print(args)

    experiment.set_name(args.task_id)
    experiment.set_cmd_args()
    experiment.log_text(args.note)

    DATA_PATH = args.input
    TRAIN_PATH = os.path.join(DATA_PATH, "train_data_train_split")
    VAL_PATH = os.path.join(DATA_PATH, "train_data_validate_split")
    TEST_PATH = os.path.join(DATA_PATH, "test_data")
    dataset_name = args.datasetname
    if dataset_name=="shanghaitech":
        print("will use shanghaitech dataset with crop ")
    elif dataset_name == "shanghaitech_keepfull":
        print("will use shanghaitech_keepfull")
    elif dataset_name == "my_bike_non_overlap":
        TRAIN_PATH = os.path.join(DATA_PATH, "train_data")
        VAL_PATH = os.path.join(DATA_PATH, "test_data")
    else:
        print("cannot detect dataset_name")
        print("current dataset_name is ", dataset_name)

    # create list
    train_list = create_image_list(TRAIN_PATH)
    val_list = create_image_list(VAL_PATH)
    test_list = create_image_list(TEST_PATH)

    # create data loader
    train_loader, train_loader_eval, val_loader, test_loader = get_dataloader(train_list, val_list, test_list, dataset_name=dataset_name, batch_size=args.batch_size,
                                                                              train_loader_for_eval_check=True,
                                                                              cache=args.cache,
                                                                              pin_memory=args.pin_memory)

    print("len train_loader ", len(train_loader))

    # model
    model_name = args.model
    experiment.log_other("model", model_name)
    experiment.add_tag(model_name)
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
    elif model_name == "BigTail4":
        model = BigTail4()
    elif model_name == "BigTail5":
        model = BigTail5()
    elif model_name == "BigTail6":
        model = BigTail6()
    elif model_name == "BigTail6i":
        model = BigTail6i()
    elif model_name == "BigTail9i":
        model = BigTail9i()
    elif model_name == "BigTail10i":
        model = BigTail10i()
    elif model_name == "BigTail11i":
        model = BigTail11i()
    elif model_name == "BigTail12i":
        model = BigTail12i()
    elif model_name == "BigTail13i":
        model = BigTail13i()
    elif model_name == "BigTail14i":
        model = BigTail14i()
    elif model_name == "BigTail15i":
        model = BigTail15i()
    elif model_name == "BigTail7":
        model = BigTail7()
    elif model_name == "BigTail8":
        model = BigTail8()
    elif model_name == "H1":
        model = H1()
    elif model_name == "H2":
        model = H2()
    elif model_name == "H3":
        model = H3()
    elif model_name == "H3i":
        model = H3i()
    elif model_name == "H4i":
        model = H4i()
    elif model_name == "H1_Bigtail3":
        model = H1_Bigtail3()
    elif model_name == "CompactCNNV7":
        model = CompactCNNV7()
    elif model_name == "CompactCNNV7i":
        model = CompactCNNV7i()
    elif model_name == "CompactCNNV8":
        model = CompactCNNV8()
    elif model_name == "CompactCNNV9":
        model = CompactCNNV9()
    elif model_name == "CSRNet":
        model = CSRNet()
    else:
        print("error: you didn't pick a model")
        sys.exit(-1)
    n_param = very_simple_param_count(model)
    experiment.log_other("n_param", n_param)
    print("n_param", n_param)
    if hasattr(model, 'model_note'):
        experiment.log_other("model_note", model.model_note)
    model = model.to(device)

    # loss function
    # loss_fn = nn.MSELoss(reduction='sum').to(device)
    if args.loss_fn == "MSE":
        loss_fn = nn.MSELoss(reduction='sum').to(device)
        print("use MSELoss")
    elif args.loss_fn == "L1":
        loss_fn = nn.L1Loss(reduction='sum').to(device)
        print("use L1Loss")
    elif args.loss_fn == "L1Mean":
        loss_fn = nn.L1Loss(reduction='mean').to(device)
        print("use L1Mean")
    elif args.loss_fn == "MSEMean":
        loss_fn = nn.MSELoss(reduction='mean').to(device)
        print("use MSEMean")
    elif args.loss_fn == "MSEL1Mean":
        loss_fn = MSEL1Loss(reduction='mean').to(device)
        print("use MSEL1Mean")
    elif args.loss_fn == "MSEL1Sum":
        loss_fn = MSEL1Loss(reduction='sum').to(device)
        print("use MSEL1Sum")
    elif args.loss_fn == "MSE4L1Mean":
        loss_fn = MSE4L1Loss(reduction='mean').to(device)
        print("use MSEL1Mean")
    elif args.loss_fn == "MSE4L1Sum":
        loss_fn = MSE4L1Loss(reduction='sum').to(device)
        print("use MSE4L1Sum")
    elif args.loss_fn == "MSENone":
        """
        Doesnt work
        because 
        RuntimeError: grad can be implicitly created only for scalar outputs
        """
        loss_fn = nn.MSELoss(reduction='none').to(device)
        print("use MSE without any reduction")
    experiment.add_tag(args.loss_fn)

    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.decay)
        print("use adam")
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    weight_decay=args.decay,
                                    momentum=args.momentum)
        print("use sgd")
    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.decay)
        print("use adamW")
    experiment.add_tag(args.optim)

    if args.lr_scheduler:
        mile_stone = lr_scheduler_milestone_builder(args.step_list, args.lr_list)
        print("lr_scheduler milestone ", mile_stone)
        lr_scheduler = PiecewiseLinear(optimizer, param_name="lr", milestones_values=mile_stone)

    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator_train = create_supervised_evaluator(model,
                                            metrics={
                                                'mae': CrowdCountingMeanAbsoluteErrorWithCount(),
                                                'mse': CrowdCountingMeanSquaredErrorWithCount(),
                                               #  'loss': Loss(loss_fn)
                                            }, device=device)

    evaluator_validate = create_supervised_evaluator(model,
                                            metrics={
                                                'mae': CrowdCountingMeanAbsoluteErrorWithCount(),
                                                'mse': CrowdCountingMeanSquaredErrorWithCount(),
                                               # 'loss': Loss(loss_fn)
                                            }, device=device)

    if args.eval_density:
        evaluator_test = create_supervised_evaluator(model,
                                                     metrics={
                                                         'ssimabs': CrowdCountingMeanSSIMabs(),
                                                         'psnrabs': CrowdCountingMeanPSNRabs(),
                                                         'ssimclamp': CrowdCountingMeanSSIMclamp(),
                                                         'psnrclamp': CrowdCountingMeanPSNRclamp(),
                                                     }, device=device)
    else:
        evaluator_test = create_supervised_evaluator(model,
                                                metrics={
                                                    'mae': CrowdCountingMeanAbsoluteErrorWithCount(),
                                                    'mse': CrowdCountingMeanSquaredErrorWithCount(),
                                                }, device=device)

    best_mae = BestMetrics(best_metric="mae")
    best_mse = BestMetrics(best_metric="mse")


    print(model)

    print(args)


    # timer
    train_timer = Timer(average=True)  # time to train whole epoch
    batch_timer = Timer(average=True)  # every batch
    evaluate_validate_timer = Timer(average=True)
    evaluate_test_timer = Timer(average=True)

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
        if args.lr_scheduler:
            to_load = {'trainer': trainer, 'model': model, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        else:
            to_load = {'trainer': trainer, 'model': model, 'optimizer': optimizer}
        checkpoint = torch.load(load_model_path)
        Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)
        print("load model complete")
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
            print("change lr to ", args.lr)
    else:
        print("do not load, keep training")

    if args.lr_scheduler:
        trainer.add_event_handler(Events.EPOCH_STARTED, lr_scheduler)

    @trainer.on(Events.ITERATION_COMPLETED(every=100))
    def log_training_loss(trainer):
        timestamp = get_readable_time()
        print(timestamp + " Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        experiment.log_metric("epoch", trainer.state.epoch)
        if not args.skip_train_eval:
            evaluator_train.run(train_loader_eval)
            metrics = evaluator_train.state.metrics
            timestamp = get_readable_time()
            print(timestamp + " Training set Results - Epoch: {}  Avg mae: {:.2f} Avg mse: {:.2f} Avg loss: {:.2f}"
                  .format(trainer.state.epoch, metrics['mae'], metrics['mse'], 0))
            # experiment.log_metric("epoch", trainer.state.epoch)
            experiment.log_metric("train_mae", metrics['mae'])
            experiment.log_metric("train_mse", metrics['mse'])
            experiment.log_metric("lr", get_lr(optimizer))

        print("batch_timer ", batch_timer.value())
        print("train_timer ", train_timer.value())
        experiment.log_metric("batch_timer", batch_timer.value())
        experiment.log_metric("train_timer", train_timer.value())

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluate_validate_timer.resume()
        evaluator_validate.run(val_loader)
        evaluate_validate_timer.pause()
        evaluate_validate_timer.step()

        metrics = evaluator_validate.state.metrics
        timestamp = get_readable_time()
        print(timestamp + " Validation set Results - Epoch: {}  Avg mae: {:.2f} Avg mse: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['mae'], metrics['mse'], 0))
        experiment.log_metric("valid_mae", metrics['mae'])
        experiment.log_metric("valid_mse", metrics['mse'])

        # timer
        experiment.log_metric("evaluate_valid_timer", evaluate_validate_timer.value())
        print("evaluate_valid_timer ", evaluate_validate_timer.value())

        # check if that validate is best
        flag_mae = best_mae.checkAndRecord(metrics['mae'], metrics['mse'])
        flag_mse = best_mse.checkAndRecord(metrics['mae'], metrics['mse'])

        if flag_mae or flag_mse:
            experiment.log_metric("valid_best_mae", metrics['mae'])
            experiment.log_metric("valid_best_mse", metrics['mse'])
            experiment.log_metric("valid_best_epoch", trainer.state.epoch)
            print("BEST VAL, evaluating on test set")
            evaluate_test_timer.resume()
            evaluator_test.run(test_loader)
            evaluate_test_timer.pause()
            evaluate_test_timer.step()
            test_metrics = evaluator_test.state.metrics
            timestamp = get_readable_time()
            print(timestamp + " Test set Results - Epoch: {}  Avg mae: {:.2f} Avg mse: {:.2f} Avg loss: {:.2f}"
                  .format(trainer.state.epoch, test_metrics['mae'], test_metrics['mse'], 0))
            experiment.log_metric("test_mae", test_metrics['mae'])
            experiment.log_metric("test_mse", test_metrics['mse'])
            experiment.log_metric("evaluate_test_timer", evaluate_test_timer.value())
            print("evaluate_test_timer ", evaluate_test_timer.value())
            # experiment.log_metric("test_loss", test_metrics['loss'])

    def checkpoint_valid_mae_score_function(engine):
        score = engine.state.metrics['mae']
        return -score


    if args.eval_only:
        print("evaluation only, no training")

        timestamp = get_readable_time()

        # if flag_mae or flag_mse:
        #     experiment.log_metric("valid_best_mae", metrics['mae'])
        #     experiment.log_metric("valid_best_mse", metrics['mse'])
        #     print("BEST VAL, evaluating on test set")
        evaluate_test_timer.resume()
        evaluator_test.run(test_loader)
        evaluate_test_timer.pause()
        evaluate_test_timer.step()
        test_metrics = evaluator_test.state.metrics
        timestamp = get_readable_time()

        if args.eval_density:
            print(timestamp + " Test set Results  ABS -  Avg ssim: {:.2f} Avg psnr: {:.2f} Avg loss: {:.2f}"
                  .format(test_metrics['ssimabs'], test_metrics['psnrabs'], 0))
            experiment.log_metric("test_ssim abs", test_metrics['ssimabs'])
            experiment.log_metric("test_psnr abs", test_metrics['psnrabs'])

            print(timestamp + " Test set Results  CLAMP -  Avg ssim: {:.2f} Avg psnr: {:.2f} Avg loss: {:.2f}"
                  .format(test_metrics['ssimclamp'], test_metrics['psnrclamp'], 0))
            experiment.log_metric("test_ssim clamp", test_metrics['ssimclamp'])
            experiment.log_metric("test_psnr clamp", test_metrics['psnrclamp'])
        else:
            print(timestamp + " Test set Results -  Avg mae: {:.2f} Avg mse: {:.2f} Avg loss: {:.2f}"
                  .format( test_metrics['mae'], test_metrics['mse'], 0))
            experiment.log_metric("test_mae", test_metrics['mae'])
            experiment.log_metric("test_mse", test_metrics['mse'])
        experiment.log_metric("evaluate_test_timer", evaluate_test_timer.value())
        print("evaluate_test_timer ", evaluate_test_timer.value())
        # experiment.log_metric("test_loss", test_metrics['loss'])
    else:
        # docs on save and load
        to_save = {'trainer': trainer, 'model': model, 'optimizer': optimizer}
        save_handler = Checkpoint(to_save, DiskSaver('saved_model/' + args.task_id, create_dir=True, atomic=True),
                                  filename_prefix=args.task_id,
                                  n_saved=3)

        save_handler_best = Checkpoint(to_save,
                                       DiskSaver('saved_model_best/' + args.task_id, create_dir=True, atomic=True),
                                       filename_prefix=args.task_id, score_name="valid_mae",
                                       score_function=checkpoint_valid_mae_score_function,
                                       n_saved=3)

        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=10), save_handler)
        evaluator_validate.add_event_handler(Events.EPOCH_COMPLETED(every=1), save_handler_best)
        trainer.run(train_loader, max_epochs=args.epochs)
