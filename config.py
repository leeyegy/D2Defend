import argparse
parser = argparse.ArgumentParser(description='Train MNIST')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--mode', default="adv", help="cln | adv")
parser.add_argument('--train_batch_size', default=50, type=int)
parser.add_argument('--test_batch_size', default=1000, type=int)

# attack
parser.add_argument("--attack_method", default="PGD", type=str,
                    choices=['FGSM', 'PGD', 'Momentum', 'STA','CW',"DeepFool","NONE",'BIM',"JSMA"])

parser.add_argument('--epsilon', type=float, default=8 / 255, help='if pd_block is used')

parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/MNIST]')

# ddid
parser.add_argument('--sigma', type=int, default=30, help='for ddid ')
parser.add_argument('--threshold', type=int, default=20, help='for DCT high-frequency exacting')
parser.add_argument('--r', type=int, default=15, help='for ddid step')
parser.add_argument('--sigma_s', type=int, default=7, help='for ddid step')
parser.add_argument('--gamma_f', type=float, nargs='+',default=[4.0,0.4,0.8], help='for ddid ')
# parser.add_argument('--adaptive', action='store_true', default=False,help='adopt adaptive sigma if true')


# net
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--num_classes', default=10, type=int)

# test
parser.add_argument('--test_samples', default=100, type=int)
parser.add_argument("--test_ssim",default=False,action='store_true')

# iteration
parser.add_argument('--nb_iters', default=3, type=int, choices=[1,2,3],help="number of iterations in ddid")

# BPDA
parser.add_argument("--max_iterations",default = 10, type =int)


args = parser.parse_args()
