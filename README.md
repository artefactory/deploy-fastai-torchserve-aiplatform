# deploy-fastai-torchserve-aiplatform

# Deploy FastAI Trained PyTorch Model in TorchServe and Host in GCP AI Platform Prediction Endpoint

- [Deploy FastAI Trained PyTorch Model in TorchServe and Host in GCP AI Platform Prediction Endpoint](#deploy-fastai-torchserve-aiplatform)
  - [Introduction](#introduction)
  - [Getting Started with A FastAI Model](#getting-started-with-a-fastai-model)
    - [Installation](#installation)
    - [Modelling](#modelling)
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
  - [Deployment to GCP AI Platform Prediction Endpoint](#deployment-to-gcp-aiplatform-prediction-endpoint)
    - [Getting Started with GCP AI Platform Prediction Endpoint](#getting-started-with-amazon-sagemaker-endpoint)
    - [Real-time Inference with Python](#real-time-inference-with-python)
  - [Conclusion](#conclusion)
  - [Reference](#reference)

## Introduction

Over the past few years, [FastAI](https://www.fast.ai/) has become one of the most cutting-edge open-source deep learning framework and the go-to choice for many machine learning use cases based on [PyTorch](https://pytorch.org/). It not only democratized deep learning and made it approachable to the general audiences, but also set as a role model on how scientific software shall be engineered, especially in Python programming. Currently, however, to deploy a FastAI model to production environment often involves setting up and self-maintaining a customized inference solution, e.g. with [Flask](https://flask.palletsprojects.com/en/1.1.x/), which is time-consuming and distracting to manage and maintain issues like security, load balancing, services orchestration, etc.

Recently, AWS developed *[TorchServe](https://github.com/pytorch/serve)* in partnership with Facebook, which is a flexible and easy-to-use open-source tool for serving PyTorch models. It removes the heavy lifting of deploying and serving PyTorch models with Kubernetes, and AWS and Facebook will maintain and continue contributing to TorchServe along with the broader PyTorch community. With TorchServe, many features are out-of-the-box and they provide full flexibility of deploying trained PyTorch models at scale so that a trained model can go to production deployment with few extra lines of code.

Meanwhile, GCP AI Platform Prediction has been a fully managed service that allows users to make real-time inferences via a REST API, and save Data Scientists and Machine Learning Engineers from managing their own server instances, load balancing, fault-tolerance, auto-scaling and model monitoring, etc. [cost-effective](https://aws.amazon.com/sagemaker/pricing/).

In this repository we demonstrate how to deploy a FastAI trained PyTorch model in TorchServe eager mode and host it in AI Platform Prediction.

## Getting Started with A FastAI Model

In this section we train a FastAI model that can solve a real-world problem with performance meeting the use-case specification. As an example, we focus on a **Scene Segmentation** use case from self-driving car.

### Installation

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

### Modelling

TO BE COMPLETED

## PyTorch Transfer Modeling from FastAI

In this section we build a pure PyTorch model and transfer the model weights from FastAI. The following materials are inspired by "[Practical-Deep-Learning-for-Coders-2.0](https://github.com/muellerzr/Practical-Deep-Learning-for-Coders-2.0/blob/master/Computer%20Vision/06_Hybridizing_Models.ipynb)" by Zachary Mueller *et al*.

### Export Model Weights from FastAI

First, restore the FastAI learner from the export pickle at the last Section, and save its model weights with PyTorch.

```python
from fastai.text import load_learner
import torch

def acc_camvid(*_): pass
def get_y(*_): pass

learn = load_learner("/home/ubuntu/.fastai/data/camvid_tiny/fastai_unet.pkl") #TODO: change model path
torch.save(learn.model.state_dict(), "fasti_unet_weights.pth")
```

It's also straightforward to obtain the FastAI prediction on a sample text.

> "2013.04 - 'Streetview of a small neighborhood', with residential buildings, Amsterdam city photo by Fons Heijnsbroek, The Netherlands" by Amsterdam free photos & pictures of the Dutch city is marked under CC0 1.0. To view the terms, visit https://creativecommons.org/licenses/cc0/1.0/

![sample_image](sample/street_view_of_a_small_neighborhood.png)

```python
text = "some random text"
pred_fastai = learn.predict(text)
pred_fastai
>>>
#TODO: put a sample prediction
```

### PyTorch Model from FastAI

Next, we need to define the model in pure PyTorch. In [Jupyter](https://jupyter.org/) notebook, one can investigate the FastAI source code by adding `??` in front of a function name. 

```python
#TODO: put code to have a pure pytorch object
```
### Weights Transfer

Now initialize the PyTorch model, load the saved model weights, and transfer that weights to the PyTorch model.

```python
model_torch_rep = DynamicUnetDIY()#TODO: adapt function name
state = torch.load("fasti_unet_weights.pth")#TODO: adapt model name
model_torch_rep.load_state_dict(state)
model_torch_rep.eval();
```

If take one sample text, transform it, and pass it to the `model_torch_rep`, we shall get an identical prediction result as FastAI's.

```python
#TODO: put inference code sample
```

Here we can see the difference: in FastAI model `fastai_unet.pkl`, it packages all the steps including the data transformation, image dimension alignment, etc.; but in `fasti_unet_weights.pth` it has only the pure weights and we have to manually re-define the data transformation procedures among others and make sure they are consistent with the training step.


## Deployment to TorchServe

In this section we deploy the PyTorch model to TorchServe. For installation, please refer to TorchServe [Github](https://github.com/pytorch/serve) Repository.

Overall, there are mainly 3 steps to use TorchServe:

1. Archive the model into `*.mar`.
2. Start the `torchserve`.
3. Call the API and get the response.

In order to archive the model, at least 2 files are needed in our case:

1. PyTorch model weights `fasti_unet_weights.pth`.
2. TorchServe custom handler.

### Custom Handler

As shown in `/deployment/handler.py`, the TorchServe handler accept `data` and `context`. In our example, we define another helper Python class with 4 instance methods to implement: `initialize`, `preprocess`, `inference` and `postprocess`.

#### `initialize`

Here we workout if GPU is available, then identify the serialized model weights file path and finally instantiate the PyTorch model and put it to evaluation mode.

```python
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
        model_dir = properties.get("model_dir")

        manifest = ctx.manifest
        logger.error(manifest)
        serialized_file = manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model definition file")

        logger.debug(model_pt_path)

        from model import DynamicUnetDIY

        state_dict = torch.load(model_pt_path, map_location=self.device)
        self.model = DynamicUnetDIY()
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.debug("Model file {0} loaded successfully".format(model_pt_path))
        self.initialized = True
```

#### `preprocess`

As described in the previous section, we re-define the image transform steps and apply them to the inference data.

```python
    def preprocess(self, data):
        """
        Scales and normalizes a PIL image for an U-net model
        """
        image = data[0].get("data")
        if image is None:
            image = data[0].get("body")

        image_transform = transforms.Compose(
            [
                transforms.Resize((96, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        image = Image.open(io.BytesIO(image)).convert(
            "RGB"
        )
        image = image_transform(image).unsqueeze_(0)
        return image
```

#### `inference`

Now convert image into PyTorch Tensor, load it into GPU if available, and pass it through the model.

```python
    def inference(self, img):
        """
        Predict the chip stack mask of an image using a trained deep learning model.
        """
        self.model.eval()
        inputs = Variable(img).to(self.device)
        outputs = self.model.forward(inputs)
        logging.debug(outputs.shape)
        return outputs
```

#### `postprocess`

Here the inference raw output is unloaded from GPU if available, and encoded with Base64 to be returned back to the API trigger.

```python
    def postprocess(self, inference_output):

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
```

Now it's ready to setup and launch TorchServe.

### TorchServe in Action

Step 1: Archive the model PyTorch

```bash
>>> torch-model-archiver --model-name fastunet --version 1.0 --model-file deployment/model.py --serialized-file model_store/fasti_unet_weights.pth --export-path model_store --handler deployment/handler.py -f
```

Step 2: Serve the Model

```bash
>>> torchserve --start --ncs --model-store model_store --models fastunet.mar
```

Step 3: Call API and Get the Response (here we use [httpie](https://httpie.org/)). For a complete response see `sample/sample_output.txt` at [here](sample/sample_output.txt).

```bash
>>> time http POST http://127.0.0.1:8080/predictions/fastunet/ @sample/street_view_of_a_small_neighborhood.png

HTTP/1.1 200
Cache-Control: no-cache; no-store, must-revalidate, private
Expires: Thu, 01 Jan 1970 00:00:00 UTC
Pragma: no-cache
connection: keep-alive
content-length: 131101
x-request-id: 96c25cb1-99c2-459e-9165-aa5ef9e3a439

{
  "base64_prediction": "GhoaGhoaGhoaGhoaGhoaGhoaGh...ERERERERERERERERERERER"
}

real    0m0.979s
user    0m0.280s
sys     0m0.039s
```

The first call would have longer latency due to model weights loading defined in `initialize`, but this will be mitigated from the second call onward. For more details about TorchServe setup and usage, please refer to `notebook/03_TorchServe.ipynb` [[link](notebook/03_TorchServe.ipynb)].

## Deployment to AI Platform Prediction Endpoint

In this section we deploy the FastAI trained Scene Segmentation PyTorch model with TorchServe in GCP AI Platform Prediction Endpoint using customized Docker image, and we will be using a `ml.g4dn.xlarge` instance. For more details about Amazon G4 Instances, please refer to [here](https://aws.amazon.com/ec2/instance-types/g4/).

### Getting Started with GCP AI Platform Prediction Endpoint







## Conclusion

This repository presented an end-to-end demonstration of deploying FastAI trained PyTorch models on TorchServe eager mode and host in GCP AI Platform Prediction. You can use this repository as a template to deploy your own FastAI models. This approach eliminates the self-maintaining effort to build and manage a customized inference server, which helps you to speed up the process from training a cutting-edge deep learning model to its online application in real-world at scale.

If you have questions please create an issue or submit Pull Request on the [GitHub](https://github.com/artefactory/deploy-fastai-torchserve-aiplatform) repository.

## Reference

- [fast.ai · Making neural nets uncool again](https://www.fast.ai/)
- [TORCHSERVE](https://pytorch.org/serve/)
- [Deploying PyTorch models for inference at scale using TorchServe host in AWS SageMaker](https://github.com/aws-samples/amazon-sagemaker-endpoint-deployment-of-fastai-model-with-torchserve)
- [Deploying PyTorch models for inference at scale using TorchServe](https://aws.amazon.com/blogs/machine-learning/deploying-pytorch-models-for-inference-at-scale-using-torchserve/)
- [Serving PyTorch models in production with the Amazon SageMaker native TorchServe integration](https://aws.amazon.com/blogs/machine-learning/serving-pytorch-models-in-production-with-the-amazon-sagemaker-native-torchserve-integration/)
- [Building, training, and deploying fastai models with Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/building-training-and-deploying-fastai-models-with-amazon-sagemaker/)
- [Running TorchServe on Amazon Elastic Kubernetes Service](https://aws.amazon.com/blogs/opensource/running-torchserve-on-amazon-elastic-kubernetes-service/)
