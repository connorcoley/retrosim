# retrosim

### summary

This repository contains the data and code needed to test a similarity-based approach to one-step retrosynthesis.

Please note that ```rdchiral``` is a work-in-progress. The current version as of June 19, 2017 has been copied into this repository for result reproducibility. An up-to-date version can be found at the public repo http://github.com/connorcoley/rdchiral

### data

The set of 50k reactions comes from http://pubs.acs.org/doi/abs/10.1021/acs.jcim.6b00564. Each reaction is pre-labeled with a class number (1-10). The dataset is further cleaned following Liu et al. (2017) (https://arxiv.org/pdf/1706.01643.pdf) so that each reaction has a single product and trivial products are excluded. Atom maps are removed for reactant atoms that do not contribute atoms to the product of interest. ```data_processed.csv``` is a Pandas dataframe and is meant to work with the functions in ```get_data.py```.

### usage

All of the "heavy lifting" occurs inside the ```scripts``` folder. ```extract_templates``` is just used for examining the templates corresponding to the training data. Likewise, ```analyze_templates``` looks at the some trends and the most common templates, but is not needed in the workflow.

After an initial data processing using ```proc_data```, the ```test_similarity``` script actually applies the similarity method using the training data as a corpus. The Jupyter notebook is meant to look at a single condition (i.e., class, fingerprint type, similarity metric) at a time. The standalone script can test the whole suite of conditions. Results are written into ```results.txt``` and are saved in separate files.

The notebook ```process_results``` reads from ```results.txt``` and examines the validation performance visually. This is how the metric was selected for use on the test data, which required a simple modification of the ```test_similarity``` script. Test results are also read using ```process_results``` and output in a tabular form at the end of the notebook.

### contact

For any questions, feel free to email ccoley@mit.edu
