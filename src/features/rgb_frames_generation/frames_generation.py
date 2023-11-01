import cv2
import numpy as np
import sys
import logging
from pathlib import Path

sys.path.append(str(Path().cwd() / 'src'))
from utils import config_logger

sys.path.append(str(Path().cwd() / 'src/configs'))
from arguments import DataArguments

logger = logging.getLogger(__name__)

def crop(image, center, radius, size=512):
    scale = 1.3
    radius_crop = (radius * scale).astype(np.int32)
    center_crop = (center).astype(np.int32)

    rect = (max(0,(center_crop-radius_crop)[0]), max(0,(center_crop-radius_crop)[1]), 
                 min(512,(center_crop+radius_crop)[0]), min(512,(center_crop+radius_crop)[1]))

    image = image[rect[1]:rect[3],rect[0]:rect[2],:]

    if image.shape[0] < image.shape[1]:
        top = abs(image.shape[0] - image.shape[1]) // 2
        bottom = abs(image.shape[0] - image.shape[1]) - top
        image = cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT,value=(0,0,0))
    elif image.shape[0] > image.shape[1]:
        left = abs(image.shape[0] - image.shape[1]) // 2
        right = abs(image.shape[0] - image.shape[1]) - left
        image = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT,value=(0,0,0))
    return image

def main():
    # Get arguments
    args = DataArguments()
    args.init_process_args()  
    args = args.parse()
    
    # Config logger
    config_logger(args.log_file)

    selected_joints = np.concatenate(([0,1,2,3,4,5,6,7,8,9,10], 
                        [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0) 
    
    input_videos_path = Path(args.videos_dir)
    if not input_videos_path.exists():
        logger.error("videos_dir does not exist")
        return
    
    input_npys_path = Path(args.keypoints_dir)
    if not input_npys_path.exists():
        logger.error("keypoints_dir does not exist")
        return
    
    output_path = Path(args.frames_dir)
    if not output_path.exists():  
          output_path.mkdir(parents=True, exist_ok=True)

    logger.info("START GENERATION!")  

    for path in input_videos_path.glob('*'):
        logger.info('----Processing {}.'.format(path))

        cap = cv2.VideoCapture(str(path))
        npy = np.load(Path(input_npys_path, '{}.npy'.format(path.stem))).astype(np.float32)
        npy = npy[:, selected_joints, :2]
        npy[:, :, 0] = 512 - npy[:, :, 0]
        xy_max = npy.max(axis=1, keepdims=False).max(axis=0, keepdims=False)
        xy_min = npy.min(axis=1, keepdims=False).min(axis=0, keepdims=False)
        assert xy_max.shape == (2,)
        xy_center = (xy_max + xy_min) / 2 - 20
        xy_radius = (xy_max - xy_center).max(axis=0)
        index = 0
        while True:
            ret, frame = cap.read()
            if ret:
                image = crop(frame, xy_center, xy_radius)
            else:
                break
            index = index + 1
            image = cv2.resize(image, (256,256))

            if not Path(output_path, path.stem).exits():
                Path(output_path, path.stem).mkdir(parents=True, exist_ok=True)
            outfile = output_path / path.stem / '{:04d}.jpg'.format(index)
            cv2.imwrite(str(outfile), image)
            logger.info("Save to {}".format(outfile))
                
                
if __name__ == '__main__':
    main()