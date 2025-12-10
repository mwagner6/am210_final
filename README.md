# Intuitive Pipeline for Causal Discovery using GDS, LiNGAM, and GRaSP

Max Wagner, for Applied Mathematics 210. 

### Installation

`pip install -r requirements.txt`

### Usage

`python pipeline.py <data path> <equivalent variance (True/False)>`

Data path must be path to a .csv file. Format should be a row of titles/headers, with each column representing a variable. There should not be an index column.

Equivalent variance should be your best guess as to whether the data have error variances that are reasonably similar to each other. Often times, but not always, this will be a reasonable assumption and not hurt model accuracy when the measurements are on the same thing (temperatures, heights of family members, etc.). 

Depending on the Gaussianity of the data and on your statements about its variance, one of three models will be used:
- GDS: Used for Gaussian data with error variances. Will recover a full causal graph, and output edge weights and error variances in results
- LiNGAM: Used for non-Gaussian data. Will recover a full causal graph, and output edge weights and error variances in results
- GRaSP: Used for mixed Gaussian and non-Gaussian data, or for unknown ratios of error variance. Will recover underlying DAG up to Markov equivalence class. Won't output edge weights or error variances