#clean
#rm features/*
#rm index/*
#rm filtering_results/*
#rm filtering_results/json/*

export PYTHONPATH=../notredame/:../featureExtraction/:../indexConstruction/:../provenanceFiltering:../provenanceGraphConstruction:../helperLibraries/:/root/faiss/:/home/jbrogan4/Documents/Projects/Medifor/faiss/

#worldIndexFile=/media/jbrogan4/scratch0/medifor/datasets/Nimble/NC2017_Dev1_Beta4/indexes/NC2017_Dev1-provenance-world.csv
#probeIndexFile=/media/jbrogan4/scratch0/medifor/datasets/Nimble/NC2017_Dev1_Beta4/indexes/NC2017_Dev1-provenance-index.csv
#nistDataDir=/media/jbrogan4/scratch0/medifor/datasets/Nimble/NC2017_Dev1_Beta4/
recall=100

worldIndexFile=../../data/testset/indexes/test-provenance-world.csv
probeIndexFile=../../data/testset/indexes/test-provenance-index.csv
nistDataDir=../../data/testset

python3.5 featureExtractionDriver.py --NISTWorldIndex $worldIndexFile --NISTDataset $nistDataDir --outputdir features/

ls -d features/* > /tmp/feature_filelist

mkdir indexTraining
mkdir index
mkdir filtering_results
mkdir filtering_results/json

#this step trains the OPQ of the index
#Set 'trainsize' (../indexConstruction/indexConstruction.py, Line 55) to at least 1000000, if you have enough features.
python3.5 indexTrainingDriver.py  --FeatureFileList feature_filelist1 --IndexOutputFile indexTraining/parameters

python3.5 indexConstructionDriver.py  --FeatureFileList feature_filelist1 --IndexOutputFile index/index --TrainedIndexParams /opt/slurm/deploy2/jbrogan/GPU_Prov_Filtering/provenance/tutorial/indexTraining/parameters

python3.5 provenanceFilteringDriver.py --NISTProbeFileList  $probeIndexFile --NISTDataset $nistDataDir --IndexOutputDir index --ProvenanceOutputFile filtering_results/results.csv --Recall  $recall

mkdir -p npz
python2 provenanceGraphDriver.py --NISTDataset $nistDataDir --FilteringResults filtering_results/results.csv --ProvenanceResultsDir graph_results/results.csv
