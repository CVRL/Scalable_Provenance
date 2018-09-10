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

worldIndexFile=/MFC18/indexes/MFC18_EvalPart1-provenancefiltering-world.csv
probeIndexFile=/opt/slurm/deploy2/jbrogan/MFC18_EvalPart1-provenance-index.csv
nistDataDir=/MFC18/

#python3.5 featureExtractionDriver.py --NISTWorldIndex $worldIndexFile --NISTDataset $nistDataDir --outputdir features/

ls -d features/* > /tmp/feature_filelist

mkdir indexTraining
mkdir index
mkdir filtering_results
mkdir filtering_results/json

#this step is optional
#python3.5 indexTrainingDriver.py  --FeatureFileList feature_filelist1 --IndexOutputFile indexTraining/parameters

python3.5 indexConstructionDriver.py  --FeatureFileList feature_filelist1 --IndexOutputFile index/index --TrainedIndexParams /opt/slurm/deploy2/jbrogan/GPU_Prov_Filtering/provenance/tutorial/indexTraining/parameters

python3.5 provenanceFilteringDriver.py --NISTProbeFileList  $probeIndexFile --NISTDataset $nistDataDir --IndexOutputDir index --ProvenanceOutputFile filtering_results/results.csv --Recall  $recall

#python3.5 provenanceFilteringDriver.py --NISTProbeFileList  /media/jbrogan4/scratch0/medifor/datasets/Nimble/NC2017_Dev1_Beta4/indexes/NC2017_Dev1-provenance-index.csv --NISTDataset   /media/jbrogan4/scratch0/medifor/datasets/Nimble/NC2017_Dev1_Beta4/ --IndexOutputDir index --ProvenanceOutputFile filtering_results/results.csv --Recall  100

mkdir -p npz
python2 provenanceGraphDriver.py --NISTDataset $nistDataDir --FilteringResults filtering_results/results.csv --ProvenanceResultsDir graph_results/results.csv
