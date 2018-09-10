#clean
#rm features/*
#rm index/*
#rm filtering_results/*
#rm filtering_results/json/*

export PYTHONPATH=../featureExtraction/:../indexConstruction/:../provenanceFiltering:../provenanceGraphConstruction:../helperLibraries/:../notredame

worldIndexFile=/mnt/datasets/development/NC2017_Dev1_Beta4/indexes/NC2017_Dev1-provenance-world.csv
probeIndexFile=/mnt/datasets/development/NC2017_Dev1_Beta4/indexes/NC2017_Dev1-provenancefiltering-index.csv
nistDataDir=/mnt/datasets/development/NC2017_Dev1_Beta4/
recall=100

python2 featureExtractionDriver.py --NISTWorldIndex $worldIndexFile --NISTDataset $nistDataDir --outputdir features/

ls -d features/* > /tmp/feature_filelist

mkdir indexTraining
mkdir index

#this step is optional
python2 indexTrainingDriver.py  --FeatureFileList /tmp/feature_filelist --IndexOutputFile indexTraining/parameters

python2 indexConstructionDriver.py  --FeatureFileList /tmp/feature_filelist --IndexOutputFile index/index --TrainedIndexParams indexTraining/parameters

python2 provenanceFilteringDriver.py --NISTProbeFileList  $probeIndexFile --NISTDataset   $nistDataDir --IndexOutputDir index --ProvenanceOutputFile filtering_results/results.csv --Recall  $recall

mkdir -p npz
python2 provenanceGraphDriver.py --NISTDataset $nistDataDir --FilteringResults filtering_results/results.csv --ProvenanceResultsDir graph_results/results.csv
