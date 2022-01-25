import sys

import Constants as C
import gif_exp_suite
from Models.microGIF import microGIF
from Models.microGIF_weights_only import microGIF_weights_only
from PDF_metrics import PDF_LFN
from TargetModels import TargetModelMicroGIF


def main(argv):
    print('Argument List:', str(argv))

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print('Using {} device'.format(device))

    # Default values
    start_seed = 42
    tar_start_seed_offset = 0
    exp_type_str = C.ExperimentType.Synthetic.name
    learn_rate = 0.015; N_exp = 2; plot_flag = True
    tau_van_rossum = 20.0
    max_train_iters = 60
    num_targets = 2
    interval_size = 1200
    batch_size = interval_size; rows_per_train_iter = interval_size
    # bin_size = int(interval_size/10)  # for RPH
    bin_size = 100  # for RPH, PNLL
    # burn_in = False
    burn_in = True

    # optimiser = 'SGD'
    optimiser = 'Adam'

    evaluate_step = 5
    data_path = None

    # loss_fn = 'BERNOULLI'
    loss_fn = 'POISSON'
    # loss_fn = None
    norm_grad_flag = False

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]
    for i, opt in enumerate(opts):
        if opt == '-h':
            print('main.py -s <script> -lr <learning-rate> -ti <training-iterations> -N <number-of-experiments> '
                  '-bs <batch-size> -tvr <van-rossum-time-constant> -rpti <rows-per-training-iteration> '
                  '-optim <optimiser> -ipr <initial-poisson-rate> -es <evaluate-step> -tmn <target-model-number> '
                  '-ss <start-seed> -et <experiment-type> -mt <model-type> -spf <silent-penalty-factor> '
                  '-ng <normalised-gradients> -dp <data-path>')
            sys.exit()
        elif opt in ("-lr", "--learning-rate"):
            learn_rate = float(args[i])
        elif opt in ("-ti", "--training-iterations"):
            max_train_iters = int(args[i])
        elif opt in ("-noe", "--numbers-of-experiments"):
            N_exp = int(args[i])
        elif opt in ("-bas", "--batch-size"):
            batch_size = int(args[i])
        elif opt in ("-bis", "--bin-size"):
            bin_size = int(args[i])
        elif opt in ("-tvr", "--van-rossum-time-constant"):
            tau_van_rossum = float(args[i])
        elif opt in ("-rpti", "--rows-per-training-iteration"):
            rows_per_train_iter = int(args[i])
        elif opt in ("-o", "--optimiser"):
            optimiser = str(args[i])
        elif opt in ("-es", "--evaluate-step"):
            evaluate_step = int(args[i])
        elif opt in ("-lfn", "--loss-function"):
            loss_fn = args[i]
        elif opt in ("-sp", "--should-plot"):
            plot_flag = bool(args[i])
        elif opt in ("-ss", "--start-seed"):
            start_seed = int(args[i])
        elif opt in ("-tsso", "--target-start-seed-offset"):
            tar_start_seed_offset = int(args[i])
        elif opt in ("-et", "--experiment-type"):
            exp_type_str = args[i]
        elif opt in ("-mt", "--model-type"):
            model_type = args[i]
        elif opt in ("-ng", "--normalised-gradients"):
            norm_grad_flag = bool(args[i])
        elif opt in ("-dp", "--data-path"):
            data_path = str(args[i])
        elif opt in ("-N", "--network-size"):
            network_size = int(args[i])
        elif opt in ("-nt", "--num-targets"):
            num_targets = int(args[i])
            assert num_targets > 0, "num targets must be >= 1. currently: {}".format(num_targets)
        elif opt in ("-bi", "--burn-in"):
            burn_in = bool(args[i])

    if loss_fn is None:
        loss_functions = [PDF_LFN.BERNOULLI, PDF_LFN.POISSON]
    else:
        loss_functions = [PDF_LFN(loss_fn).name]

    if exp_type_str in [C.ExperimentType.Synthetic.name, C.ExperimentType.SanityCheck.name]:
        for loss_fn in loss_functions:
            for f_i in range(3+tar_start_seed_offset, 3+tar_start_seed_offset+num_targets):
                target_model_name = 'gif_soft_continuous_populations_model{}'.format(f_i)
                # pop_sizes, target_model = TargetModelMicroGIF.micro_gif_populations_model_full_size(random_seed=f_i)
                pop_sizes, target_model = TargetModelMicroGIF.get_low_dim_micro_GIF_transposed(random_seed=f_i)

                constants = C.Constants(learn_rate=learn_rate, train_iters=max_train_iters, N_exp=N_exp, batch_size=batch_size,
                                        tau_van_rossum=tau_van_rossum, rows_per_train_iter=rows_per_train_iter, optimiser=optimiser,
                                        initial_poisson_rate=0., loss_fn=loss_fn, evaluate_step=evaluate_step,
                                        plot_flag=plot_flag, start_seed=start_seed, target_fname=target_model_name,
                                        exp_type_str=exp_type_str, silent_penalty_factor=None,
                                        norm_grad_flag=norm_grad_flag, data_path=data_path, bin_size=bin_size,
                                        burn_in=burn_in, tar_start_seed_offset=tar_start_seed_offset)

                gif_exp_suite.start_exp(constants=constants, model_class=microGIF_weights_only, target_model=target_model, pop_sizes=pop_sizes)


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
