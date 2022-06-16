
This code forked form the code from the blog post [🦄 How to build a State-of-the-Art Conversational AI with Transfer Learning](https://medium.com/@Thomwolf/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313).

## Installation

To install and use the training and inference scripts please clone the repo and install the requirements:

```bash
git clone https://github.com/huggingface/transfer-learning-conv-ai
cd transfer-learning-conv-ai
pip install -r requirements.txt
python -m spacy download en
```

## Training Config

I ran this on a Vertex AI Notebook with Pytorch preinstalled.
XLA allows for running on TPU, but I just ran the training on CPU
```
export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0" XRT_WORKERS="localservice:0;grpc://localhost:51011"
```

Finally run the training. 
```
python ./train_on_bucket.py
```

## TODO
parameterize bucket name instead of hard coding
use bigquery to which training sets were used on each model
