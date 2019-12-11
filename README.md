## Comparing the SVM and RVM 

This contains the fully-reproducible work to provide a simple comparison of the Support Vector Machine (SVM) and Relevance Vector Machine (RVM) on a simple dataset – the Wisconsin Breast Cancer Diagnosis dataset. 

`bootstrap_loop.py` contains the code to run the full set of bootstrapped simulations. From the command line, run `time python3 bootstrap_loop.py > log.txt`. ` svm_rvm_helpers.py` contains helper functions used in the `bootstrap_loop.py` program. The detailed output is logged in `log.text` and a summary table is saved in `results_svm-vs-rvm.csv`. 