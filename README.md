# Deploy FastAI Trained PyTorch Model in TorchServe and Host in GCP AI Platform Prediction

- [Deploy FastAI Trained PyTorch Model in TorchServe and Host in GCP AI Platform Prediction](#deploy-fastai-torchserve-aiplatform)
  - [Introduction](#introduction)
  - [Getting Started with A FastAI Model](#getting-started-with-a-fastai-model)
    - [Installation](#installation)
  - [PyTorch Transfer Modeling from FastAI](#pytorch-transfer-modeling-from-fastai)
    - [Export Model Weights from FastAI](#export-model-weights-from-fastai)
    - [PyTorch Model from FastAI Source Code](#pytorch-model-from-fastai-source-code)
    - [Weights Transfer](#weights-transfer)
  - [Deployment to TorchServe](#deployment-to-torchserve)
    - [Custom Handler](#custom-handler)
      - [`initialize`](#initialize)
      - [`preprocess`](#preprocess)
      - [`inference`](#inference)
      - [`postprocess`](#postprocess)
    - [TorchServe in Action](#torchserve-in-action)
  - [Deployment to GCP AI Platform Prediction](#deployment-to-gcp-aiplatform-prediction)
    - [Getting Started with GCP AI Platform Prediction](#getting-started-with-gcp-aiplatform-prediction)
    - [Real-time Inference with Python](#real-time-inference-with-python)
  - [Conclusion](#conclusion)
  - [Reference](#reference)

## Introduction

Over the past few years, NLP has had a huge hype of transfer learning. [FastAI](https://www.fast.ai/) is a deep learning library which provides practitioners with high-level components that can quickly and easily provide state-of-the-art results in standard deep learning domains.. It has become one of the most cutting-edge open-source deep learning framework and the go-to choice for many machine learning use cases based on [PyTorch](https://pytorch.org/).

Recently, AWS developed *[TorchServe](https://github.com/pytorch/serve)* in partnership with Facebook. TorchServe makes it easy to deploy PyTorch models at scale in production environments. It removes the heavy lifting of developing your own client server architecture.

Meanwhile, GCP AI Platform Prediction has been a fully managed  [cost-effective](https://cloud.google.com/ai-platform/prediction/pricing) service that allows users to make real-time inferences via a REST API, and save Data Scientists and Machine Learning Engineers from managing their own server instances.

In this repository we demonstrate how to deploy a FastAI trained PyTorch model in TorchServe eager mode and host it in AI Platform Prediction.


## 1 - Installation

The first step is to install FastAI package, which is covered in its [Github](https://github.com/fastai/fastai) repository.

> If you're using Anaconda then run:
> ```python
> conda install -c fastai -c pytorch -c anaconda fastai gh anaconda
> ```
> ...or if you're using miniconda) then run:
> ```python
> conda install -c fastai -c pytorch fastai
> ```

For other installation options, please refer to the FastAI documentation.

## 2 - Reusing fastai model in pytorch

### Export Model Weights from FastAI

First, restore the FastAI learner from the export pickle at the last Section, and save its model weights with PyTorch.

#### Text version
```python
from fastai.text import load_learner
from fastai.text.learner import _get_text_vocab, get_c

import torch

VOCABZ_SZ = len(_get_text_vocab(dls)) #dls is the dataloader you used for training
N_CLASSES = get_c(dls)
CONFIG = awd_lstm_clas_config.copy()

learn = load_learner("fastai_cls.pkl")
torch.save(learn.model.state_dict(), "fastai_cls_weights.pth")

text = "This was a very good movie"
pred_fastai = learn.predict(text)
pred_fastai
>>>
(Category tensor(1), tensor(1), tensor([0.0036, 0.9964]))
```

Once you got the model weights, put the .pth file in model/text or model/image.

#### Image version
```python
from fastai.vision.all import get_c

IMAGE_SIZE = dls.one_batch()[0].shape[-2:] #dls is the dataloader you used for training
N_CLASSES = get_c(dls)

image_path = "street_view_of_a_small_neighborhood.png"
pred_fastai = learn.predict(image_path)
pred_fastai[0].numpy()
>>>
array([[26, 26, 26, ...,  4,  4,  4],
       [26, 26, 26, ...,  4,  4,  4],
       [26, 26, 26, ...,  4,  4,  4],
       ...,
       [17, 17, 17, ..., 30, 30, 30],
       [17, 17, 17, ..., 30, 30, 30],
       [17, 17, 17, ..., 30, 30, 30]])
```
### PyTorch Model from FastAI

Next, we need to define the model in pure PyTorch. In [Jupyter](https://jupyter.org/) notebook, one can investigate the FastAI source code by adding `??` in front of a function name. 
#### Text version
```python
from fastai.text.models.core import get_text_classifier
from fastai.text.all import AWD_LSTM

model_torch = get_text_classifier(AWD_LSTM, VOCABZ_SZ, N_CLASSES, config=CONFIG)
```

The important thing here is that get_text_classifier fastai function outputs a torch.nn.modules.module.Module which therefore is a pure PyTorch object.

#### Image version

```python
from fastai.vision.learner import create_unet_model
from fastai.vision.models import resnet50


model_torch = create_unet_model(resnet50, N_CLASSES, IMAGE_SIZE)
```
### Weights Transfer

Now initialize the PyTorch model, load the saved model weights, and transfer that weights to the PyTorch model.

```python
state = torch.load("fastai_cls_weights.pth")
model_torch.load_state_dict(state)
model_torch.eval()
```

If we take one sample, transform it, and pass it to the `model_torch`, we shall get an identical prediction result as FastAI's.

### Preprocessing inputs

#### Text version
```python
import torch
from fastai.text.core import Tokenizer, SpacyTokenizer
from fastai.text.data import Numericalize

example = "Hello, this is a test."

tokenizer = Tokenizer(
    tok=SpacyTokenizer("en")
)
numericalizer = Numericalize(vocab=vocab)

example_processed = numericalizer(tokenizer(example))
>>>
tensor([ 4,  7, 26, 29, 16, 72, 69, 31])

inputs = example_processed.resize(1, len(example_processed))
outputs = model_torch.forward(inputs)[0]
preds = torch.softmax(outputs, dim=-1) #You can use any activation function you need
>>>
tensor([[0.0036, 0.9964]], grad_fn=<SoftmaxBackward>)
```
#### Image version

```python
from torchvision import transforms
from PIL import Image
import numpy as np

image_path = "street_view_of_a_small_neighborhood.png"

image = Image.open(image_path).convert("RGB")
image_tfm = transforms.Compose(
    [
        transforms.Resize((96, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

x = image_tfm(image).unsqueeze_(0)

# inference on CPU
raw_out = model_torch_rep(x)
raw_out.shape
>>> torch.Size([1, 32, 96, 128])

pred_res = raw_out[0].argmax(dim=0).numpy().astype(np.uint8)
pred_res
>>>
array([[26, 26, 26, ...,  4,  4,  4],
       [26, 26, 26, ...,  4,  4,  4],
       [26, 26, 26, ...,  4,  4,  4],
       ...,
       [17, 17, 17, ..., 30, 30, 30],
       [17, 17, 17, ..., 30, 30, 30],
       [17, 17, 17, ..., 30, 30, 30]], dtype=uint8)
```

## 3-  Deployment to TorchServe

In this section we deploy the PyTorch model to TorchServe. For installation, please refer to TorchServe [Github](https://github.com/pytorch/serve) Repository.

Overall, there are mainly 3 steps to use TorchServe:

1. Archive the model into `*.mar`.
2. Start the `torchserve`.
3. Call the API and get the response.

In order to archive the model, at least 2 files are needed in our case:

1. PyTorch model weights `fastai_cls_weights.pth`.
2. TorchServe custom handler.

### Custom Handler

As shown in `/deployment/handler.py`, the TorchServe handler accept `data` and `context`. In our example, we define another helper Python class with 4 instance methods to implement: `initialize`, `preprocess`, `inference` and `postprocess`.

Now it's ready to setup and launch TorchServe.

### TorchServe in Action

Step 1: Archive the model PyTorch

```bash
>>> torch-model-archiver \
  --model-name=fastai_model \
  --version=1.0 \
  --serialized-file=/home/model-server/fastai_v2_cls_weights.pth \
  --extra-files=/home/model-server/config.py,/home/model-server/vocab.json
  --handler=/home/model-server/handler.py \
  --export-path=/home/model-server/model-store/
```

Step 2: Serve the Model

```bash
>>> torchserve --start --ncs --model-store model_store --models fastai_model.mar
```

Step 3: Call API and Get the Response (here we use [httpie](https://httpie.org/)). For a complete response see `sample/sample_output.txt` at [here](sample/sample_output.txt).

```bash
>>> time http POST http://127.0.0.1:8080/predictions/fastai_model/ @input.txt

HTTP/1.1 200
Cache-Control: no-cache; no-store, must-revalidate, private
Expires: Thu, 01 Jan 1970 00:00:00 UTC
Pragma: no-cache
connection: keep-alive
content-length: 131101
x-request-id: 96c25cb1-99c2-459e-9165-aa5ef9e3a439

{
  "Categories": "1",
  "Tensor": [
    0.0036,
    0.9964
  ]
}

real    0m0.979s
user    0m0.280s
sys     0m0.039s
```

The first call would have longer latency due to model weights loading defined in `initialize`, but this will be mitigated from the second call onward.

## 4- Deployment to AI Platform Prediction

In this section we deploy the FastAI trained Scene Segmentation PyTorch model with TorchServe in GCP AI Platform Prediction using customized Docker image. or more details about GCP AI Platform Prediction routines using custom containers please refer to [here](https://cloud.google.com/ai-platform/prediction/docs/use-custom-container).

### Getting Started with GCP AI Platform Prediction

There are 3 steps to host a model on AI Platform Prediction with TorchServe:

1. Create a new model on AI Platform. Be careful to choose a model with a regional endpoint to have the option to use a custom container.
```bash
gcloud beta ai-platform models create $MODEL_NAME \
  --region=$REGION \
  --enable-logging \
  --enable-console-logging
```
2. Build a docker image of the torchserve API package and push it to a container registry following the [custom container requirements](https://cloud.google.com/ai-platform/prediction/docs/custom-container-requirements). Be careful to create a regional artifact registry repository if it's doesn't already exist
```bash
gcloud beta artifacts repositories create $ARTIFACT_REGISTRY_NAME \
 --repository-format=docker \
 --location=$REGION
 ```
```bash
 docker build -t $REGION-docker.pkg.dev/$GCP_PROJECT_ID/$ARTIFACT_REGISTRY_NAME/$DOCKER_NAME:$DOCKER_TAG . -f TextDockerfile
```
```bash
 docker push $REGION-docker.pkg.dev/$GCP_PROJECT_ID/$ARTIFACT_REGISTRY_NAME/$DOCKER_NAME:$DOCKER_TAG
```

3. [Optional] Run your docker locally and try to send a prediction

```bash
docker run -d -p 8080:8080 --name local_imdb  $REGION-docker.pkg.dev/$GCP_PROJECT_ID/$ARTIFACT_REGISTRY_NAME/$DOCKER_NAME:$DOCKER_TAG
```
```bash
curl -X POST -H "Content-Type: application/json" -d '["this was a bad movie"]' 127.0.0.1:8080/predictions/fastai_model
```
If everything is working okay, you should receive a response from the server in your console
```bash
[
  {
    "Categories": "0",
    "Tensor": [
      0.9999990463256836,
      9.371918849865324e-07
    ]
  }
]
```

4. Create AI Platform version model using the docker image of the torchserve API package.
```bash
gcloud beta ai-platform versions create $VERSION_NAME  --region=$REGION --model=$MODEL_NAME --image=$REGION-docker.pkg.dev/$GCP_PROJECT_ID/$ARTIFACT_REGISTRY_NAME/$DOCKER_NAME:$DOCKER_TAG  --ports=8080   --health-route=/ping   --predict-route=/predictions/fastai_model
```


## Conclusion

This repository presented an end-to-end demonstration of deploying a FastAI text classifier model on GCP AI Platform Prediction. It allows a user to serve a fastai model without going through the self-maintaining effort to build and manage a customized inference server. This repository was inspired by another project that aimed to deploy a fastai image classifier on AWS SageMaker Inference Endpoint [here](https://github.com/aws-samples/amazon-sagemaker-endpoint-deployment-of-fastai-model-with-torchserve).

If you have questions please create an issue or submit Pull Request on the [GitHub](https://github.com/artefactory/deploy-fastai-torchserve-aiplatform) repository.

## Reference

- [fast.ai · Making neural nets uncool again](https://www.fast.ai/)
- [TORCHSERVE](https://pytorch.org/serve/)
- [Deploying PyTorch models for inference at scale using TorchServe host in AWS SageMaker](https://github.com/aws-samples/amazon-sagemaker-endpoint-deployment-of-fastai-model-with-torchserve)
- [Deploying PyTorch models for inference at scale using TorchServe](https://aws.amazon.com/blogs/machine-learning/deploying-pytorch-models-for-inference-at-scale-using-torchserve/)
- [Serving PyTorch models in production with the Amazon SageMaker native TorchServe integration](https://aws.amazon.com/blogs/machine-learning/serving-pytorch-models-in-production-with-the-amazon-sagemaker-native-torchserve-integration/)
- [Building, training, and deploying fastai models with Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/building-training-and-deploying-fastai-models-with-amazon-sagemaker/)
- [Running TorchServe on Amazon Elastic Kubernetes Service](https://aws.amazon.com/blogs/opensource/running-torchserve-on-amazon-elastic-kubernetes-service/)
