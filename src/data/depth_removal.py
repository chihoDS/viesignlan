import sys
import logging
from pathlib import Path

sys.path.append(str(Path().cwd() / 'src'))
from utils import config_logger

sys.path.append(str(Path().cwd() / 'src/configs'))
from arguments import DataArguments

logger = logging.getLogger(__name__)

def main():
    # Get arguments
    args = DataArguments()
    args.init_removal_args()  
    args = args.parse()
    
    # Config logger
    config_logger(args.log_file)
    
    dir_path = Path(args.depth_videos_input)
    
    # Check whether dir_path exists or not
    if not dir_path.exists():
        logger.error('Input directory does not existed.')
        return
    
    # dir_path exists
    logger.info('Remove depth videos in {}.'.format(args.depth_videos_input))
    
    # Use the glob method to get a list of all file paths in the directory
    file_paths = list(dir_path.glob('*.mp4'))
    remove_list = []
    
    for path in file_paths:
        # If it is depth video
        if "depth" in str(path):
            remove_list.append(path)
            path.unlink()
    logger.info('Number of videos deleted: {}.'.format(len(remove_list)))
    logger.info("Samples of removed videos' file path: {}.".format(remove_list[:3]))
    
if __name__ == '__main__':
    main()


    