# Human Re-Identification

This repo contains the training code for a model that can generate discriminative image embeddings for the human re-identification task on the [Market-1501](https://paperswithcode.com/dataset/market-1501) dataset.

## Query image
![Query image](./assets/query_image.png "Query image")

| Retrieval position | Gallery image                           | Confidence score (0-1) |
| ------------------ | --------------------------------------- | ---------------------- |
| 1st                | ![1st result](./assets/top1_9410.png)   | 0.9410                 |
| 5th                | ![5th result](./assets/top5_8607.png)   | 0.8607                 |
| 10th               | ![10th result](./assets/top10_8130.png) | 0.8130                 |
| 15th               | ![15th result](./assets/top15_7678.png) | 0.7678                 |

The results above were produced from the pre-trained model located at `models/masked-hog-908/best.pth`. This is the default model used for inference and is evaluated in [Results](#results).

# Environment setup
This project has only been tested on Python 3.12.4, so it's the recommended version.

There are three options for setting up your development environment

1. [Pull Docker image](#pull-image-from-docker-hub) (quickest way to play with the model)
2. [Virtual environment](#initializing-a-virtual-env) (best for training new models)
3. [Build Docker image](#build-docker-image-from-source)

## Pull image from Docker Hub

Pull the pre-built image from docker hub
```bash
docker pull texashookem/human_reid_mi:inference
```

Feel free to restrict compute resources to the container, but for demonstration purposes the container should be launched in privileged mode to access needed compute.
This will launch the container in interactive mode and initialize a bash shell at startup.

```bash
docker run --privileged -it texashookem/human_reid_mi:inference
```
Perform inference on the default query image 

```bash
python inference.py --load_index
```

Perform inference on list of query images 

```bash
python inference.py --load_index --query_paths <list of image paths> --topk 10
```


## Initializing a virtual env
  Ensure python 3.12 is installed

   1. Create a python virtual environment running `python -m venv venv`
   2. Activate the virtual environment `source venv/bin/activate`
   3. Install project dependencies `pip install -r requirements.txt`
   4. Download Market-1501 dataset and rearrange image directories by running `python data_prep.py`

## Build Docker image from source

1. Build docker container 
```bash
 docker build -t reid .
```
This will create a container with the target python version, project dependencies installed, and the data set already present.

2. Run container in interactive mode
```bash
docker run -it -p 6006:6006 -p 8080:8080 reid
```

3. Test default model
```bash
python inference.py --load_index
```

# Project setup
This section describes the project's experimentation workflow and is meant for those interested in training additional models or
reproducing the results presented. If you wish to make use of the provided model weights, skip to [Inference](#inference).

The project utilizes Tensorboard for tracking experiment metrics and MlFlow for tracking experiment configuration.
These tools are recommended for training new models as they centralize experiment information and aid in experiment analysis. Each experiment is assigned a name from MlFlow, so the experiment name is used as an id to create and access files within the `/models` and `/index` directories.
 To start the Tensorboard and MlFlow dashboards run the following script on the command line.
 ```bash 
 ./dashboards.sh
 ```
   This script is responsible for managing the visualization dashboards. It will automatically stop all launched services when it's terminated. Once the script executes, Tensorboard will be accessible at [http://localhost/6006](http://localhost:6006) and MlFlow on [http://localhost:8080](http:/localhost:8080). At this point, you're in a good spot to begin training new models or evaluating the provided models. A recommended experimentation workflow is outlined below

1. Train a new model by running `python train.py`. See the script source for available cli parameters.
2. [Evaluate](#evaluation) the trained model on the gallery set by running `python eval.py --exp_name <name> --save_index`
3. [View Top-5](#inference) retrieval results for a query image by running `python inference.py --exp_name <name> --query_paths <img path> --load_index`

Utilizing a trained model within the `eval.py` and `inference.py` scripts requires creating an index over the gallery to perform image retrieval. This is a highly computational process, so it's recommended to serialize an index for a specific model by passing the `--save_index` argument to either of these scripts. This will serialize index data to disk for later use when the `--load_index` argument is provided to the above-mentioned scripts.

# Inference

To perform inference on an arbritrary set of query images, pass the query image path along with an experiment name as arguments to `inference.py`. This script will return the top 5, by default, most similar images to the query in the gallery along with their cosine similarity scores.


Some example use cases are provided below.

To retrieve the top 5 most similar images with the default model for a single image
```bash
python inference.py --query_paths ./data/market-1501-grouped/query/0003/0003_c1s6_015971_00.jpg --load_index
```

To retrieve the top 5 most similar images with the default model on all images for identity 0003
```bash
python inference.py --query_paths ./data/market-1501-grouped/query/0003/* --load_index
```

To retrieve the top 5 most similar images with the default model for all query images
```bash
python inference.py --query_paths ./data/market-1501-grouped/query/*/* --load_index
```

To retrieve the top 20 most similar images with the default model on the default query image
```bash
python inference.py --load_index --topk 20
```

To retrieve the top 5 most similar images with a new model on the default image and save the internal index for evaluation
```bash
python inference.py --exp_name <mlflow exp name> --save_index
```

# Evaluation

The following snippets will help you evaluate the model on the Market-1501 dataset by computing these
evaluation metrics: R-1, R-5, R-10, mAP. 

To compute evaluation metrics for the default model

```bash
python eval.py --load_index
```

To compute evaluation metrics on a new model and save the index for inference
```bash
python eval.py --exp_name <exp name in /runs> --save_index
```

# Results

I fine-tuned a pre-trained [DenseNet](https://arxiv.org/abs/1608.06993) model on a seperate classification layer without any data augmentation or learning rate scheduling. I used SGD to optimize model parameters and trained for a total of 20 epochs on an NVIDIA GeForce RTX 4060 Ti. The table below compares my model's performance - /models/masked-hog-908/best.pth - to competitive ReId models.

| Model     | Learning rate | Epochs | Batch size | Rank-1 Accuracy | Rank-5 Accuracy | Rank-10 Accuracy | mAP       |
| --------- | ------------- | ------ | ---------- | --------------- | --------------- | ---------------- | --------- |
| Mine      | 0.02          | 20     | 16         | 84.68%          | 66.89%          | 47.77%           | 65.34%    |
| stReID    | 0.10          | 60     | 32         | **97.2%**       | **99.3%**       | **99.5%**        | 86.7%     |
| PDF       | 0.01          | n/a    | 16         | 84.14%          | n/a             | n/a              | 64.41%    |
| TransReID | 0.008         | n/a    | 64         | 95.2%           | n/a             | n/a              | **89.5%** |

Although my model is not on par with newer models like stReID or TransReID, it obtained decent retrieval performance within 10 epochs of training and became competitive to [PDF](https://paperswithcode.com/paper/pose-driven-deep-convolutional-model-for) after 20 epochs.