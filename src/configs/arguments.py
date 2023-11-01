import argparse


class DataArguments():
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            conflict_handler="resolve"
        )
        self.init_general_args()

    def init_general_args(self) -> None:
        """

        
        """   
        self.parser.add_argument("--log_file", type=str, default=None, help="path to logging file txt which store logged information")
    
    def init_removal_args(self) -> None:
        """

        
        """      
        self.parser.add_argument("--depth_videos_input", type=str, default="data/raw/AUTSL/train", help="") 
    
    def init_process_args(self) -> None:
        """
 
        
        """   
        self.parser.add_argument("--keypoints_cfg", type=str, default="src/configs/wholebody_w48_384x288.yaml", help="")
        self.parser.add_argument("--pretrained_keypoints_generation_model", type=str, default="models/Wholepose/hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth", help="")
        self.parser.add_argument("--features_cfg", type=str, default="src/configs/wholebody_w48_384x384_adam_lr1e-3.yaml", help="")
        self.parser.add_argument("--pretrained_features_extraction_model", type=str, default="models/Wholepose/wholebody_hrnet_w48_384x384.pth", help="")
        
        self.parser.add_argument("--videos_dir", type=str, default="data/raw/AUTSL/train", help="")
        self.parser.add_argument("--keypoints_dir", type=str, default="data/interim/AUTSL/keypoints/train", help="")
        self.parser.add_argument("--frames_dir", type=str, default="data/interim/AUTSL/frames/train", help="")
        self.parser.add_argument("--features_dir",type=str, default="data/interim/AUTSL/features/train", help="path to output feature dataset")
        self.parser.add_argument("--flow_dir",type=str, default="data/interim/AUTSL/flow/train", help="path to output flow dataset")
        
        self.parser.add_argument("--multi_scales", type=list, default=[512,640], help="")
        self.parser.add_argument("--keypoints_resolution", type=int, default=512, help="")
        self.parser.add_argument("--features_resolution", type=int, default=384, help="")

        self.parser.add_argument("--istrain", type=bool, default=False, help="generate training data or not")
        
        
    def parse(self) -> object:
        args = self.parser.parse_args()
        return args
    

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

class SLGCNAguments():
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            conflict_handler="resolve"
        )
        self.init_general_args()

    def init_general_args(self) -> None:
        """

        
        """   
        self.parser.add_argument("--log_file", type=str, default=None, help="path to logging file txt which store logged information")
        self.parser.add_argument('--work-dir', default='./work_dir/temp', help='the work folder for storing results')

        self.parser.add_argument('-model_saved_name', default='')
        self.parser.add_argument('-Experiment_name', default='')
        self.parser.add_argument('--config', default='./config/nturgbd-cross-view/test_bone.yaml', help='path to the configuration file')
    
    def init_processor_args(self) -> None:
        self.parser.add_argument('--phase', default='train', help='must be train or test')
        self.parser.add_argument('--save-score', type=str2bool, default=False, help='if ture, the classification score will be stored')

    def init_visulize_debug_args(self) -> None:
        self.parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
        self.parser.add_argument('--log-interval', type=int, default=100, help='the interval for printing messages (#iteration)')
        self.parser.add_argument('--save-interval', type=int, default=2, help='the interval for storing models (#iteration)')
        self.parser.add_argument('--eval-interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
        self.parser.add_argument('--print-log', type=str2bool, default=True, help='print logging or not')
        self.parser.add_argument('--show-topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        
    def init_feeder_args(self) -> None:
        self.parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
        self.parser.add_argument('--num-worker', type=int, default=32,help='the number of worker for data loader')
        self.parser.add_argument('--train-feeder-args', default=dict(), help='the arguments of data loader for training')
        self.parser.add_argument('--test-feeder-args', default=dict(), help='the arguments of data loader for test')
        
    def init_model_args(self) -> None:    
        self.parser.add_argument('--model', default=None, help='the model will be used')
        self.parser.add_argument('--model-args', type=dict, default=dict(), help='the arguments of model')
        self.parser.add_argument('--weights', default=None, help='the weights for network initialization')
        self.parser.add_argument('--ignore-weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')

    def init_optim_args(self) -> None:
        self.parser.add_argument('--base-lr', type=float, default=0.01, help='initial learning rate')
        self.parser.add_argument('--step', type=int, default=[20, 40, 60], nargs='+', help='the epoch where optimizer reduce the learning rate')
        self.parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
        self.parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        self.parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
        self.parser.add_argument('--batch-size', type=int, default=256, help='training batch size')
        self.parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
        self.parser.add_argument('--start-epoch', type=int, default=0, help='start training from which epoch')
        self.parser.add_argument('--num-epoch', type=int, default=80, help='stop training in which epoch')
        self.parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay for optimizer')
        self.parser.add_argument('--keep_rate', type=float, default=0.9, help='keep probability for drop')
        self.parser.add_argument('--groups', type=int, default=8, help='decouple groups')
        self.parser.add_argument('--only_train_part', default=True)
        self.parser.add_argument('--only_train_epoch', default=0)
        self.parser.add_argument('--warm_up_epoch', default=0)

    def parse(self) -> object:
        args = self.parser.parse_args()
        return args