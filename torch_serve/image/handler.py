import logging
import os
import sys

sys.path.insert(0, os.path.abspath('.'))

import base64
import io

import numpy as np
import torch
from fastai.vision.learner import create_unet_model
from fastai.vision.models import resnet50
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from config import IMAGE_SIZE, N_CLASSES

logger = logging.getLogger(__name__)


def find_max_list(mylist):
    list_len = [len(i) for i in mylist]
    return max(list_len)


class ImageClassifierHandler:
    """
    ImageClassifierHandler handler class.
    """

    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False

    def initialize(self, ctx):
        """
        load eager mode state_dict based model
        """
        properties = ctx.system_properties
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )

        logger.info(f"Device on initialization is: {self.device}")

        manifest = ctx.manifest
        logger.info(manifest)
        serialized_file = manifest["model"]["serializedFile"]

        if not os.path.isfile(serialized_file):
            raise RuntimeError("Missing the model definition file")

        logger.debug(serialized_file)

        state_dict = torch.load(serialized_file, map_location=self.device)
        self.model = create_unet_model(resnet50, N_CLASSES, IMAGE_SIZE)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.debug(f"Model file {serialized_file} loaded successfully")
        self.initialized = True

    @staticmethod    
    def preprocess(data):
        """
        Scales and normalizes a PIL image for an U-net model
        """
        image = data[0].get("data")
        if image is None:
            image = data[0].get("body")

        image_transform = transforms.Compose(
            [
                # must be consistent with model training
                transforms.Resize((96, 128)),
                transforms.ToTensor(),
                # default statistics from imagenet
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        image = Image.open(io.BytesIO(image)).convert(
            "RGB"
        )  # in case of an alpha channel
        image = image_transform(image).unsqueeze_(0)
        return image

    def inference(self, img):
        """
        Predict the chip stack mask of an image using a trained deep learning model.
        """
        logger.info(f"Device on inference is: {self.device}")
        self.model.eval()
        inputs = Variable(img).to(self.device)
        outputs = self.model.forward(inputs)
        logging.debug(outputs.shape)
        return outputs

    @staticmethod
    def postprocess(inference_output):

        if torch.cuda.is_available():
            inference_output = inference_output[0].argmax(dim=0).cpu()
        else:
            inference_output = inference_output[0].argmax(dim=0)

        return [
            {
                "base64_prediction": base64.b64encode(
                    inference_output.numpy().astype(np.uint8)
                ).decode("utf-8")
            }
        ]


_service = ImageClassifierHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data
