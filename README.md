# Neural Networks for Graph-Structured Time Series with Application to Seismic Data

Supplementary material for the master thesis project in Environmental Modelling by Jonas Moritz Zender, submitted on the 07.05.2025.

For a detailed explanation of the dataset, experiments, and results, refer to the [thesis pdf file](master_thesis_zender_jonas_moritz_6410897.pdf).

## Graph INSTANCE Dataset

This work is based on a processed subset of the INSTANCE dataset originally published by _Michelini et al. (2021)_:

 - Michelini, A., Cianetti, S., Gaviano, S., Giunchi, C., Jozinović, D., and Lauciani,
   V. (2021). INSTANCE – the Italian seismic dataset for machine learning. Earth
   System Science Data, 13(12):5509–5544.

The corresponding paper can be found [here](https://essd.copernicus.org/articles/13/5509/2021/) and the dataset can be downloaded from [here](https://www.pi.ingv.it/banche-dati/instance/). You need to download the single HDF5 File in ground motion units containing the time series data, and the 'Events metadata version 3' csv file.

The dataset used here - for simplicity referred to as Graph INSTANCE (GI) dataset in my thesis - is derived from the INSTANCE dataset by:

 - filtering the original data for earthquakes events which
   - have a magnitude of 3.0 or greater
   - have at least 10 stations that recorded the event
   - belong to the IV network
 - calculating distances between the filtered stations, to be used for calculating edge weights in graphs
 - creating training, validation and test splits, stratified w.r.t. to earthquake magnitude
 - a Pytorch dataset class, which reads and processes the data and  dynamically creates graphs for each sample. The dataset class is described in the section below.

For reproducing the steps to derive the Graph INSTANCE dataset from the original INSTANCE dataset, use the [preprocessing notebook](GraphINSTANCE/process_instance_dataset.ipynb). All necessary files, like dataset splits, used in this project are provided under the [GraphINSTANCE](GraphINSTANCE) directory. You may still use the notebook to filter the HDF5 file and reduce the original file size of the INSTANCE dataset from ~156GB to ~20GB, and to filter the csv file containing the metadata.

⚠️ __Note__: This repository does __not__ claim ownership of the original __INSTANCE__ dataset. Full credit goes to _Michelini et al. (2021)_ for the creation and publication of the original data.

## Code

### PyTorch Dataset Class

The [graph_instance.py](src/data/graph_instance.py) scripts contains a number of relevant classes in Pytorch and Pytorch Geometric to read the Graph INSTANCE dataset. 

The basic Dataset classes are:

 - ```GIDataset```: Basic Pytorch dataset class, responsible for loading the dataset, building graphs, processing of the time series, export of the graphs to WKT format, and more. The output of this is not directly suited for batching in a DataLoader.
 - ```GIDatasetPyG```: Wrapper for the ```GIDataset``` class, which converts its outputs into a batchable ```torch_geometric.data.Data``` object, that can be used in a DataLoader.

For _self-supervised pretraining_ using the Peak Ground Acceleration (PGA) as target in random time windows, use:

 - ```GIDatasetPretrain```: PyTorch dataset class, that modifies the basic ```GIDataset``` for the pretraining objective; mainly by adjusting the trimming of the time series and calculating the PGA within the random time window.
 - ```GIDatasetPretrainPyG```: Wrapper for the ```GIDatasetPretrain``` class, which converts its outputs into a batchable ```torch_geometric.data.Data``` object, that can be used in a DataLoader. 

For adding _artificial nodes_ to the graph to get predictions for locations where there is no station in the graph, e.g. to generate earthquake intensity maps, use the following classes:
 - ```GIDatasetPyG_with_artificial_node```: PyTorch Geometric dataset class, that modifies the basic ```GIDataset``` to include an artificial node at a desired location.
 - ```ArtificialNodeIterator```: Iterator, that first creates a grid of latitude-longitude coordinate pairs, based on a desired grid size, for a given sample from the Graph INSTANCE dataset, and then iteratively returns ```GIDatasetPyG_with_artificial_node``` objects, each with an artificially added node with a coordinate pair from the previously generated grid.

For in-depth information, refer to the documentation of the Python script and the thesis PDF file, describing the steps in much more detail.

### Training Scripts

 - [Experiment 1 (Learning Rate and Epicenter)](src/training_scripts/exp1_lr_and_epicenter): Training the baseline model with different learning rates on the main task (Intensity Measurements) and auxiliary task (earthquake epicenter)
 - [Experiment 2 (Edge Filtering)](src/training_scripts/exp2_edge_filtering): Testing different scenarios for filtering edges (minimum node degree, maximum graph diameter)
 - [Experiment 3 (Raw TS Tokens)](src/training_scripts/exp3_raw_ts_tokens): Investigate whether raw time series tokens in the GNN improve the model
 - [Experiment 4 (Static Features)](src/training_scripts/exp4_static_features): Influence of static features and small architectural changes
 - [Experiment 5 (Pretraining and Station Dropout)](src/training_scripts/exp5_pretraining_and_station_dropout): Self-supervised pretraining, finetuning (with and without freezing the base), and finetuning without pretraining
 - [Experiment 6 (Probabilistic Models, Mixture Models and Deep Ensembles)](src/training_scripts/exp6_probabilistic_models_and_ensembles): Training of probabilistic models (predicting mean and variance), mixture models composed of probabilistic models, and deep ensembles, trained end-to-end using probabilistic models

### Utilities

 - [Utilities for gathering model outputs](src/utils/gather_model_outputs.py): Utilities to gather model outputs for bootstrapping, using different time series paddings, station dropout ratios etc.
 - [Bootstrapping Utilities](src/utils/bootstrapping.py): utility functions for generating or loading bootstrapping sample ids, and for applying bootstrapping to individual models, or whole folders.
