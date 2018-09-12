# Scalable Image Provenance Library #

![The full image provenance pipeline](figure1.png?raw=true "Figure1")

DESCRIPTION:
************
This code implements the algorithms described in ["Image Provenance at Scale"](https://arxiv.org/abs/1801.06510)

As expressed in the above illustration, it includes 2 main parts:
* [Image Filtering](image_filtering/README.md)
* [Graph Building](graph_building/README.md)

Each part of this pipeline creates output that is then used as input for the next part of the pipeline. 

OVERVIEW:
*********
Image extraction utilizes the Distributed SURF method to extract 5000 local features from each image.  

The image index utilizes OPQ, which must be trained on a subset of these features before index construction can begin

The index is built using an Inverted File binned on the trained OPQ centroids using [FAISS](https://github.com/facebookresearch/faiss).

Filtering is a 2 query for each image that returns a JSON list of image ranks from the index

Graph building takes the JSON list and constructs a final provenance graph using the algorithm illustrated above.

REQUIREMENTS:
*************
Python2.7

Python3.6

numpy

opencv

faiss

scipy

scikit-image

matplotlib

psutil

progressbar2

urlparse

joblib


QUESTIONS:
**********
Please contact Joel Brogan (joel.r.brogan.20@nd.edu ), Aparna Bharati (aparna.bharati.1@nd.edu), or Daniel Moreira (dhenriq1@nd.edu).
