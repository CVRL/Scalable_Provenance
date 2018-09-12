# Image Provenance Analysis at Scale
## Provenance Graph Building
![The full image provenance pipeline](figure1.png?raw=true "Figure1")

This code implements the algorithms described in ["Image Provenance at Scale"](https://arxiv.org/abs/1801.06510), with respect to the step of "graph building" (right side of the illustration above).

### Language
* Python 2.7

### API Dependencies
* numpy
* opencv 2.4

### Content
* src
Source content.
Auxiliary libs are stored in "lib".
Main programs are numbered in the way they might be executed for a complete provenance graph building:
- 01_build_adj_mat.py: Builds the adjacency (dissimilarity) matrices and saves them in a proper ".npz" file.
- 02_join_adj_mat.py: Joins/combines the adjacency matrices that are saved in a given ".npz" file.
- 03_build_graph.py: Builds the provenance DAG from the given adjacency matrix.
  
* script  
NIST-Nimble-based execution shell scripts.
The scripts are numbered in the way they might be executed for a complete provenance graph building.

### Questions
Please contact Aparna Bharati (aparna.bharati.1@nd.edu) or Daniel Moreira (daniel.moreira.comp@gmail.com).
