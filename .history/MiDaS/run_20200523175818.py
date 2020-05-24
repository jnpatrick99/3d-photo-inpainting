"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
from . import utils
import cv2
import numpy as np

from torchvision.transforms import Compose
from .models.midas_net import MidasNet
from .models.transforms import Resize, NormalizeImage, PrepareForNet


def run_depth(img_names, input_path, output_path, model_path):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda")
    print("device: %s" % device)

    # load network
    model = MidasNet(model_path, non_negative=True)

    transform = Compose(
        [
            Resize(
                1536.,
                1536.,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    model.to(device)
    model.eval()

    # get input
    # img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")

    for ind, img_name in enumerate(img_names):

        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))


        # input

        img = utils.read_image(img_name)
        img_input = transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        # output
        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
        )
        print( " size: {} ",img.shape);
        print("output: {}",filename)
        utils.write_depth(filename, prediction, bits=2)


        w = img.shape[1]
        scale = 640. / max(img.shape[0], img.shape[1])
        target_height, target_width = int(round(img.shape[0] * scale)), int(round(img.shape[1] * scale))        
        print( " out size: {} ",prediction.shape);
        print( " resize: {} ",target_width,target_height);
        prediction = utils.resize_depth2(prediction, target_width, target_height)



        np.save(filename + '.npy', prediction)

    print("finished")


if __name__ == "__main__":
    # set paths
    INPUT_PATH = "input"
    OUTPUT_PATH = "output"
    # MODEL_PATH = "model.pt"
    MODEL_PATH = "model.pt"

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(INPUT_PATH, OUTPUT_PATH, MODEL_PATH)