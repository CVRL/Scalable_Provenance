# Scalable Image Provenance Library #

![The full image provenance pipeline](figure1.png?raw=true "Figure1")

DESCRIPTION:
************
This code implements the algorithms described in ["Image Provenance at Scale"](https://arxiv.org/abs/1801.06510)

It includes 4 main parts:
* Image feature extraction
* Image search index construction
* Query image filtering
* Provenance Graph Building

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


USAGE:
******
To run the example test case, run ./provenance/tutorial/runPython3.sh NOTE: This is a test case with 100 world images and 10 probe images, and will not produce any meaningful results!

DOCKER:
******
A dockerfile has also been included which contains the exact environmental setup required for this code to run. You can either build the dockerfile yourself, or use the dockerfile as a template to set up the code requirements lcoally.