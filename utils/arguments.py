import argparse
import datetime


class Arguments():
    def __init__(self):
        self.parser  = argparse.ArgumentParser()

    def get_training_args(self):

        self.parser.add_argument('--batch_size', type=int, default=64, help="Batch size while evaluating")
        self.parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs")

        self.parser.add_argument('--init', type=str, default='xavier', choices=["xavier", "gaussian"], help="Network initializer")

        self.parser.add_argument('--regularizer_type', type=str, default='none', help="Type of the regularizer. Default: Do not use. ")
        self.parser.add_argument('--regularizer_weight', type=float, default = 0, help="Regularizer weight")

        self.parser.add_argument('--optim_name', type=str, default='sgd', choices=["adam", "sgd", "adagrad", "rmsprop"], help="Which optimizer to use.")
        self.parser.add_argument('--lr', type=float, default=0.001, help="Learning rate. If learning rate scheduler is provided, this is the initial learning rate.")
        self.parser.add_argument('--lr_sched', type=str, choices=['none', 'exp', 'step', 'custom'], default='none', help="Scheduler type. none for no schedule. ")
        self.parser.add_argument('--exp_decay_rate', type=float, default=0.8, help="Decay rate for exponential lr scheduler.")
        self.parser.add_argument('--exp_decay_steps', type=int, default=1000, help="Decay step for exponential lr scheduler.")


        self.parser.add_argument('--sgd_momentum', type=float, default=0.9, help="SGD momentum hyperparameter")
        self.parser.add_argument('--adam_beta1', type=float, default=0.9, help="Beta1 Parameter of Adam")
        self.parser.add_argument('--adam_beta2', type=float, default=0.99, help="Beta2 Parameter of Adam")

        self.parser.add_argument('--experiment_name', '-n', required=True, help="Name of the experiment. Experiment name is used to keep results directory organized.")

        self.parser.add_argument('--bn', action="store_true", default=False, help="Whether to use Batch Normalization")
        self.parser.add_argument('--dropout', action="store_true", default=False, help="Whether to use Dropout")
        self.parser.add_argument('--drop_rate', type=float, default=0.5, help="Dropout Rate")
        self.parser.add_argument('--kernel_l2', type=float, default=0.01, help="kernel l2 parameter for l2 regularizers")
        self.parser.add_argument('--bias_l2', type=float, default=0.00, help="bias l2 parameter for l2 regularizers")

        self.parser.add_argument('--network_type', type=str, default="vanilla", choices=["vanilla"], help="The network architectute. Default: vanilla -> CNNs, MaxPooling, FCs. Use it to create another model like transformer")
        self.parser.add_argument('--vanilla_conv_count', type=int, default=0, help="Number of conv layers in the intermediate layers ")
        self.parser.add_argument('--loss_function', type=str, default="mse", choices=["mse", "mae", "rmse", "combined"], help="Loss function for the training ")
        self.parser.add_argument('--patience', type=int, default=3, help="Number of epoch for early stopping ")
        self.parser.add_argument('--patience_start', type=int, default=30, help="Start for checking early stopping ")

        args = self.parser.parse_args()

        current_time = datetime.datetime.now().strftime("%b%d-%H:%M")
        args.experiment_name = f"{args.experiment_name}/{current_time}"

        return args

    def get_valid_args(self):
        self.parser.add_argument('--batch_size', type=int, default=64, help="Batch size while evaluating")

        self.parser.add_argument('--ckpt_dir', required=True, help="Model that will be evaluated")

        self.parser.add_argument('--experiment_name', '-n', required=True, help="Name of the experiment. Experiment name is used to keep results directory organized.")
        
        args = self.parser.parse_args()

        current_time = datetime.datetime.now().strftime("%b%d-%H:%M")
        args.experiment_name = f"{args.experiment_name}/{current_time}"

        return args