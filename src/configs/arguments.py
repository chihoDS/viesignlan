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
        self.parser.add_argument("--cfg_filename", type=str, default="src/features/skeleton_keypoints_generation/wholebody_w48_384x288.yaml", help="")
        self.parser.add_argument("--pretrained_whole_body_pose_model", type=str, default="models/Wholepose/hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth", help="")
        
        self.parser.add_argument("--input_videos", type=str, default="data/raw/AUTSL/train", help="")
        self.parser.add_argument("--output_keypoints", type=str, default="data/interim/AUTSL/keypoints/train", help="")
        self.parser.add_argument("--output_frames", type=str, default="data/interim/AUTSL/frames/train", help="")
        
        self.parser.add_argument("--multi_scales", type=list, default=[512,640], help="")
        self.parser.add_argument("--resolution", type=int, default=512, help="")
        
        
        
        self.parser.add_argument("--skeleton_features_input_path", type=str, default="data/raw/AUTSL/train", help="")
        self.parser.add_argument("--skeleton_features_output_path", type=str, default="data/processed/AUTSL/skeleton_features/train", help="")
        
        self.parser.add_argument("--frames_input_path", type=str, default="data/raw/AUTSL/train", help="")
        self.parser.add_argument("--f_feoutput_path", type=str, default="data/processed/AUTSL/skeleton_features/train", help="")
        
        self.parser.add_argument("--process_all", action='store_true', help="process all the data or just part of it")
        self.parser.add_argument("--from_index", type=int, default=None)
        self.parser.add_argument("--to_index", type=int, default=None)
        self.parser.add_argument("--base_url", type=str, default = "https://www.youtube.com/watch?v=", help="base url to add id")
        self.parser.add_argument("--fps", type = int, default=1, help="frame per second")
        
        
    def parse(self) -> object:
        args = self.parser.parse_args()
        return args
    