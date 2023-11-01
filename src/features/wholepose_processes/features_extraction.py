import cv2
import numpy as np
import sys
import logging
from pathlib import Path

import torch


from pose_hrnet import get_pose_net
from torch.autograd import Variable

from config import cfg
from default import _C as cfg

from pose_hrnet import get_pose_net

sys.path.append(str(Path().cwd() / 'src'))
from utils import config_logger

sys.path.append(str(Path().cwd() / 'src/configs'))
from arguments import DataArguments

logger = logging.getLogger(__name__)

means=[0.485, 0.456, 0.406]
stds=[0.229, 0.224, 0.225]


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
    
    output_path = Path(args.features_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        cfg.merge_from_file(args.features_cfg)
        device = torch.device("cuda")

        model = get_pose_net(cfg, is_train=False)
        checkpoint = torch.load(args.pretrained_features_extraction_model, map_location="cuda:0")
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()

        logger.info("START EXTRACTION!")

        for path in input_path.glob('*'):
            outfile = output_path / '{}.pt'.format(path.stem)

            if Path(outfile).exists():
                continue

            logger.info('----Processing {}.'.format(path))

            frames = []
            frames_flip = []
            cap = cv2.VideoCapture(str(path))

            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            logger.info("Length: {} | FPS: {}".format(length, fps))

            index = 0
            space = 0
            num_frame = 0

            while cap.isOpened():
                success, image = cap.read()
                if success:
                    num_frame += 1
                else:
                    logger.error("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break
                    # continue

                image = cv2.resize(image,(args.features_resolution,args.features_resolution))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_flip = cv2.flip(image,1)
                image = image.astype(np.float32) / 255.
                image_flip = image_flip.astype(np.float32) / 255.
                
                for i in range(3):
                    image[:, :, i] = image[:, :, i] - means[i]
                    image[:, :, i] = image[:, :, i] / stds[i]
                    image_flip[:, :, i] = image_flip[:, :, i] - means[i]
                    image_flip[:, :, i] = image_flip[:, :, i] / stds[i]
                image = image.transpose((2, 0, 1))
                image_flip = image_flip.transpose((2,0,1))

                if length < 60:
                    num_to_repeat = int(60/length)
                    space = 1
                    if 60-length*num_to_repeat>0:
                        space =int(length/(60-length*num_to_repeat))
                    else:
                        space = 100000
                    if index % space == 0 and index < length - (60 %(60-length)) and space < 60:
                        num_to_repeat += 1
                    for i in range(num_to_repeat):
                        frames.append(image)
                        frames_flip.append(image_flip)
                    index += 1
                    if num_frame == length:
                        for i in range(60-len(frames)):
                            frames.append(image)
                            frames_flip.append(image_flip)
                        break
                elif length == 60:
                    frames.append(image)
                    frames_flip.append(image_flip)
                else:
                    space = int(length/(length-60))
                    if index % space == 0 and index < length - (length % (length-60)):
                        index += 1
                        continue
                    index += 1
                    frames.append(image)
                    frames_flip.append(image_flip)

            data = np.array(frames)
            input = Variable(torch.from_numpy(data).cuda())
            out = model(input)
            m = torch.nn.MaxPool2d(3, stride=2,padding=1)
            out = m(out)
            out = m(out)
            selected_indices = [0,71,77,85,89,5,6,7,8,9,10,91,93,95,96,99,100,103,104,107,108,111,112,114,116,117,120,121,124,125,128,129,132]
            newout = out[:,selected_indices,:,:]
            newout = newout.view(1,-1,24,24)
            torch.save(newout,outfile)
            logger.info('Save to {}.'.format(outfile))

            if args.istrain:
                data = np.array(frames_flip)
                input = Variable(torch.from_numpy(data).cuda())
                out = model(input)
                out = m(out)
                out = m(out)
                newout = out[:,selected_indices,:,:]
                newout = newout.view(1,-1,24,24)
                outfile = output_path / '{}_flip.pt'.format(path.stem)
                torch.save(newout,outfile)

            if len(frames)!=60:
                break
            cap.release()

if __name__ == '__main__':
    main()