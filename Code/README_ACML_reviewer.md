Hi Dear reviewer,
We have conducted all our experiments via SLURM, to reproduce the xact results you can runmeta_test.py orexperiment_meta_testing.sh on slurm
Here is the codebase to reproduce our experiments. 
1. At first we have CLAMS in gama folder, this constructs the pipeliens for every dataset
2. similarity.sh this computes similarity for every dataset
3. in the meta_test.py file we already provide computed distance and algorithm data
