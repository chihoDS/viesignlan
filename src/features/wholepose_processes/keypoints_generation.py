import sys
import logging
from pathlib import Path


import torch
import torch.nn.functional as f

import numpy as np
import cv2
from collections import OrderedDict



from pose_hrnet import get_pose_net

sys.path.append(str(Path().cwd() / 'src'))
from utils import pose_process, config_logger, get_all_filepaths_in_dir, get_all_filenames_in_dir

sys.path.append(str(Path().cwd() / 'src/configs'))
from arguments import DataArguments

from default import _C as cfg

logger = logging.getLogger(__name__)

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

index_mirror = np.concatenate([
                [1,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16],
                [21,22,23,18,19,20],
                np.arange(40,23,-1), np.arange(50,40,-1),
                np.arange(51,55), np.arange(59,54,-1),
                [69,68,67,66,71,70], [63,62,61,60,65,64],
                np.arange(78,71,-1), np.arange(83,78,-1),
                [88,87,86,85,84,91,90,89],
                np.arange(113,134), np.arange(92,113)
                ]) - 1
assert(index_mirror.shape[0] == 133)

def norm_numpy_totensor(img):
    img = img.astype(np.float32) / 255.0
    for i in range(3):
        img[:, :, :, i] = (img[:, :, :, i] - mean[i]) / std[i]
    return torch.from_numpy(img).permute(0, 3, 1, 2)
def stack_flip(img):
    img_flip = cv2.flip(img, 1)
    return np.stack([img, img_flip], axis=0)

def merge_hm(hms_list):
    assert isinstance(hms_list, list)
    for hms in hms_list:
        hms[1,:,:,:] = torch.flip(hms[1,index_mirror,:,:], [2])
    
    hm = torch.cat(hms_list, dim=0)
    hm = torch.mean(hms, dim=0)
    return hm

def main():
    # Get arguments
    args = DataArguments()
    args.init_process_args()  
    args = args.parse()
    
    # Config logger
    config_logger(args.log_file)

    input_path = Path(args.videos_dir)
    if not input_path.exists():
        logger.error('videos_dir is not existed.')
        return
    
    # paths = get_all_filepaths_in_dir(input_path, '*.mp4')
    # names = get_all_filenames_in_dir(input_path, '*.mp4')
    # logger.info('Number of videos: {}.'.format(len(paths)))
    
    output_path = Path(args.keypoints_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():   
        cfg.merge_from_file(args.keypoints_cfg)
        
        newmodel = get_pose_net(cfg, is_train=False)    
        checkpoint = torch.load(args.pretrained_keypoints_generation_model)

        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        
        for k, v in state_dict.items():
            if 'backbone.' in k:
                name = k[9:] # remove module.
            if 'keypoint_head.' in k:
                name = k[14:] # remove module.
            new_state_dict[name] = v
            
        newmodel.load_state_dict(new_state_dict)
        newmodel.cuda().eval()      

        logger.info("START GENERATION!")  

        for path in input_path.glob('*'):
            outfile = output_path / '{}.npy'.format(path.stem)

            if Path(outfile).exists():
                continue
            
            logger.info('----Processing {}.'.format(path))
            
            # Create a VideoCapture object to access the video source
            cap = cv2.VideoCapture(str(path))

            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            logger.info('Frame: {}x{}.'.format(frame_width, frame_height))

            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            logger.info("Length: {} | FPS: {}".format(length, fps))
            

            output_list = []
            
            # Continuously loop as long as the video capture (cap) is open and operational.
            while cap.isOpened():
                success, img = cap.read()
                if not success:
                    logger.error("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break
                    # continue
                frame_height, frame_width = img.shape[:2]
                img = cv2.flip(img, flipCode=1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                out = []
                for scale in args.multi_scales:
                    if scale != args.keypoints_resolution:
                        img_temp = cv2.resize(img, (scale,scale))
                    else:
                        img_temp = img
                    img_temp = stack_flip(img_temp)
                    img_temp = norm_numpy_totensor(img_temp).cuda()
                    hms = newmodel(img_temp)
                    
                    if scale != args.keypoints_resolution:
                        out.append(f.interpolate(hms, (frame_width // 4,frame_height // 4), mode='bilinear'))
                    else:
                        out.append(hms)

                out = merge_hm(out)
                result = out.reshape((133,-1))
                result = torch.argmax(result, dim=1)
                result = result.cpu().numpy().squeeze()

                y = result // (frame_width // 4)
                x = result % (frame_width // 4)
                pred = np.zeros((133, 3), dtype=np.float32)
                pred[:, 0] = x
                pred[:, 1] = y

                hm = out.cpu().numpy().reshape((133, frame_height//4, frame_height//4))

                pred = pose_process(pred, hm)
                pred[:,:2] *= 4.0 
                assert pred.shape == (133, 3)
                output_list.append(pred)
            
            logger.info('Save to {}.'.format(outfile))

            output_list = np.array(output_list)
            np.save(outfile, output_list)
            cap.release()

if __name__ == '__main__':
    main()