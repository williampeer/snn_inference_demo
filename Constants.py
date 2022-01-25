import enum

from torch import optim

import IO


class Constants:
    def __init__(self, learn_rate, train_iters, N_exp, batch_size, tau_van_rossum,
                 initial_poisson_rate, rows_per_train_iter, optimiser, loss_fn, evaluate_step,
                 data_set=None, data_path=None, plot_flag=True, fitted_model_path=None, data_bin_size=None,
                 target_bin_size=None, start_seed=0, target_fname=None, exp_type_str=None, silent_penalty_factor=None,
                 norm_grad_flag=False, bin_size=400, burn_in=False, tar_start_seed_offset=0):
        if data_bin_size is not None:
            self.data_bin_size = int(data_bin_size)
        else:
            self.data_bin_size = None

        if target_bin_size is not None:
            self.target_bin_size = int(target_bin_size)
        else:
            self.target_bin_size = None

        self.data_set = data_set
        self.data_path = data_path
        self.fitted_model_path = fitted_model_path

        self.learn_rate = float(learn_rate)
        self.train_iters = int(train_iters)
        self.N_exp = int(N_exp)
        self.batch_size = int(batch_size)
        self.tau_van_rossum = float(tau_van_rossum)
        self.initial_poisson_rate = float(initial_poisson_rate)
        self.rows_per_train_iter = int(rows_per_train_iter)
        self.loss_fn = loss_fn
        self.evaluate_step = evaluate_step
        self.plot_flag = plot_flag
        self.start_seed = start_seed
        self.tar_start_seed_offset = tar_start_seed_offset
        self.target_fname = target_fname
        self.bin_size = bin_size
        self.burn_in = burn_in
        try:
            self.EXP_TYPE = ExperimentType[exp_type_str]
        except:
            raise NotImplementedError('ExperimentType not found.')
        self.silent_penalty_factor = silent_penalty_factor
        self.norm_grad_flag = norm_grad_flag

        # self.UUID = uuid.uuid4().__str__()
        self.UUID = IO.dt_descriptor()

        self.optimiser = None
        if optimiser == 'SGD':
            self.optimiser = optim.SGD
        elif optimiser == 'Adam':
            self.optimiser = optim.Adam
        elif optimiser == 'RMSprop':
            self.optimiser = optim.RMSprop
        else:
            raise NotImplementedError('Optimiser not supported. Please use either SGD or Adam.')

    def __str__(self):
        return 'data_bin_size: {}, target_bin_size: {}, learn_rate: {}, train_iters: {}, N_exp: {}, batch_size: {},' \
               'tau_van_rossum: {}, initial_poisson_rate: {}, rows_per_train_iter: {}, ' \
               'optimiser: {}, loss_fn: {}, data_set: {}, evaluate_step: {}, fitted_model_path: {}, data_path: {}, ' \
               'plot_flag: {}, start_seed: {}, target_fname: {}, EXP_TYPE: {}, silent_penalty_factor: {}, ' \
               'norm_grad_flag: {}, bin_size: {}, burn_in: {}, tar_start_seed_offset: {}'.\
            format(self.data_bin_size, self.target_bin_size, self.learn_rate, self.train_iters, self.N_exp,
                   self.batch_size, self.tau_van_rossum, self.initial_poisson_rate, self.rows_per_train_iter,
                   self.optimiser, self.loss_fn, self.data_set, self.evaluate_step, self.fitted_model_path,
                   self.data_path, self.plot_flag, self.start_seed, self.target_fname, self.EXP_TYPE,
                   self.silent_penalty_factor, self.norm_grad_flag, self.bin_size, self.burn_in, self.tar_start_seed_offset)


class ExperimentType(enum.Enum):
    DataDriven = 1
    Synthetic = 2
    SanityCheck = 3
