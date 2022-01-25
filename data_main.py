import sys

import Constants as C
import data_exp_suite
from Models.GLIF import GLIF
from Models.LIF import LIF
from Models.LIF_R import LIF_R
from Models.LIF_R_ASC import LIF_R_ASC


def main(argv):
    print('Argument List:', str(argv))

    # Default values
    start_seed = 0
    exp_type_str = C.ExperimentType.DataDriven.name
    learn_rate = 0.03; N_exp = 5; tau_van_rossum = 100.0; plot_flag = True

    max_train_iters = 20; batch_size = 400; rows_per_train_iter = 4000
    loss_fn = None

    optimiser = 'Adam'
    # optimiser = 'SGD'
    initial_poisson_rate = 10.  # Hz

    evaluate_step = 2
    prefix = '/home/user/data/some_data/'
    data_path = prefix + 'experiment_42.mat'
    model_type = None

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]
    for i, opt in enumerate(opts):
        if opt == '-h':
            print('main.py -s <script> -lr <learning-rate> -ti <training-iterations> -N <number-of-experiments> '
                  '-bs <batch-size> -tvr <van-rossum-time-constant> -rpti <rows-per-training-iteration> '
                  '-optim <optimiser> -ipr <initial-poisson-rate> -es <evaluate-step> -tmn <target-model-number> '
                  '-trn <target-rate-number>')
            sys.exit()
        elif opt in ("-lr", "--learning-rate"):
            learn_rate = float(args[i])
        elif opt in ("-ti", "--training-iterations"):
            max_train_iters = int(args[i])
        elif opt in ("-N", "--numbers-of-experiments"):
            N_exp = int(args[i])
        elif opt in ("-bs", "--batch-size"):
            batch_size = int(args[i])
        elif opt in ("-tvr", "--van-rossum-time-constant"):
            tau_van_rossum = float(args[i])
        elif opt in ("-rpti", "--rows-per-training-iteration"):
            rows_per_train_iter = int(args[i])
        elif opt in ("-o", "--optimiser"):
            optimiser = str(args[i])
        elif opt in ("-ipr", "--initial-poisson-rate"):
            initial_poisson_rate = float(args[i])
        elif opt in ("-es", "--evaluate-step"):
            evaluate_step = int(args[i])
        elif opt in ("-lfn", "--loss-function"):
            loss_fn = args[i]
        elif opt in ("-sp", "--should-plot"):
            plot_flag = bool(args[i])
        elif opt in ("-trn", "--target-rate-number"):
            trn = int(args[i])
        elif opt in ("-ss", "--start-seed"):
            start_seed = int(args[i])
        elif opt in ("-et", "--experiment-type"):
            exp_type_str = args[i]
        elif opt in ("-mt", "--model-type"):
            model_type = args[i]

    all_models = [LIF, GLIF, GLIF_dynamic_R_I, LIF_R, LIF_ASC, LIF_R_ASC]
    models = [LIF, LIF_R, LIF_ASC, LIF_R_ASC, GLIF, GLIF_dynamic_R_I]
    if loss_fn is None:
        loss_functions = ['frd', 'vrd', 'frdvrd', 'frdvrda']
    else:
        loss_functions = [loss_fn]
    if model_type is not None and model_type in str(all_models):
        for m in all_models:
            if m.__name__ == model_type:
                models = [m]
        if len(models) > 1:
            print('Did not find supplied model type. Iterating over all implemented models..')

    for m_class in models:
        for loss_fn in loss_functions:
            constants = C.Constants(learn_rate=learn_rate, train_iters=max_train_iters, N_exp=N_exp, batch_size=batch_size,
                                    tau_van_rossum=tau_van_rossum, rows_per_train_iter=rows_per_train_iter, optimiser=optimiser,
                                    initial_poisson_rate=initial_poisson_rate, loss_fn=loss_fn, evaluate_step=evaluate_step,
                                    plot_flag=plot_flag, start_seed=start_seed, target_fname='data', exp_type_str=exp_type_str,
                                    data_path=data_path)

            data_exp_suite.start_exp(constants=constants, model_class=m_class)


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
