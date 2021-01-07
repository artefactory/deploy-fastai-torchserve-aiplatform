import logging
import os
import sys

import numpy as np
import torch
from fastai.text.data import SPProcessor
from fastai.text.learner import get_text_classifier
from fastai.text.models.awd_lstm import AWD_LSTM

from config import (CONFIG_DICT, CLASSES, THRESHOLD_PREDICTION,
                    VOCAB_SIZE)

sys.path.insert(0, os.path.abspath('.'))


logger = logging.getLogger(__name__)


def find_max_list(mylist):
    list_len = [len(i) for i in mylist]
    return max(list_len)


class TextClassifierHandler:
    """
    TextClassifierHandler handler class.
    """

    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.preprocessor = None
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
        self._load_preprocessor()

        if not os.path.isfile(serialized_file):
            raise RuntimeError("Missing the model definition file")

        logger.debug(serialized_file)

        state_dict = torch.load(serialized_file, map_location=self.device)
        self.model = get_text_classifier(
            AWD_LSTM,
            VOCAB_SIZE,
            len(CLASSES),
            config=CONFIG_DICT)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.debug(f"Model file {serialized_file} loaded successfully")
        self.initialized = True

    def _load_preprocessor(self):
        preprocessor = SPProcessor(
            sp_model="spm.model",
            sp_vocab="spm.vocab")
        self.preprocessor = preprocessor

    def preprocess(self, data):
        logger.debug(f"input data : {data}")
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        logger.debug(f"text : {text} of type {type(text)}")
        if not isinstance(text, list):
            text = [text]
        logger.debug(f"text : {text}")
        text_preprocessed = [self.preprocessor.process_one(el) for el in text]
        max_len = find_max_list(text_preprocessed)
        text_preprocessed = [
            np.pad(el, (0, max_len - len(el)), 'constant', constant_values=(None, 0))
            for el in text_preprocessed]
        x_tensor = torch.LongTensor(text_preprocessed)
        logger.debug(f"x_tensor : {x_tensor}")
        return x_tensor

    def inference(self, txt, activation=(lambda x: torch.softmax(x, dim=-1)):
        """
        Predict the chip stack mask of an image using a trained deep learning model.
        """
        logger.info(f"Device on inference is: {self.device}")
        self.model.eval()
        inputs = torch.autograd.Variable(txt).to(self.device)
        outputs = self.model.forward(inputs)[0]
        logger.debug(outputs.shape)
        return activation(outputs)

    def postprocess(self, inference_result):
        result = []
        for tensor in inference_result:
            class_assigned = ",".join(
                [
                    CLASSES[label]
                    for label in range(len(CLASSES)) if tensor[label] > THRESHOLD_PREDICTION
                ]
            )
            result.append(
                {
                    "Categories": class_assigned,
                    "Tensor": tensor.detach().numpy().astype(np.float32).tolist()
                }
            )
        logger.debug(f"result : {result}")
        return [result]


_service = TextClassifierHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data