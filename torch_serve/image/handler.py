import logging
import os
import sys

import numpy as np
import torch
from torchvision import transforms

from fastai.vision.learner import create_cnn_model
from fastai.vision.models import resnet50

from config import CLASSES

sys.path.insert(0, os.path.abspath('.'))


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
        self.model = create_cnn_model(resnet50, len(CLASSES))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.debug(f"Model file {serialized_file} loaded successfully")
        self.initialized = True

    def preprocess(self, data):
        image_tfm = transforms.Compose(
            [
                transforms.Resize((96, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        x_tensor = image_tfm(data).unsqueeze_(0)
        return x_tensor

    def inference(self, image, activation=(lambda x: x)):
        """
        Predict the chip stack mask of an image using a trained deep learning model.
        """
        logger.info(f"Device on inference is: {self.device}")
        self.model.eval()
        inputs = torch.autograd.Variable(image).to(self.device)
        outputs = self.model.forward(inputs)
        logger.debug(outputs.shape)
        return activation(outputs)

    @staticmethod
    def postprocess(inference_result):
        result = []
        for tensor in inference_result:
            class_assigned = CLASSES[tensor.argmax()]
            result.append(
                {
                    "Categories": str(class_assigned),
                    "Tensor": tensor.detach().numpy().astype(np.float32).tolist()
                }
            )
        logger.debug(f"result : {result}")
        return [result]


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
