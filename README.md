To setup:
* Create a [new conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
* Activate new conda environment
* Download requirements.txt file (`pip install -r requirements.txt`)
* Download [DELPHI](https://github.com/lucian-ilie/DELPHI) & point its top directory to this one
    * (For the sake of scrapiness actually already have it downloaded so skip this step, we'll move it out of this directory later)

To run:
* `python src/runner.py` 


TODOs:
* (Yehlin?) Add the volcano plots with markers of our predicted peptides into F1 notebook
* (Yitong) Generate list of 100 peptides which we most strongly predict for 12ca5 & MDM2 to F3
* Benchmark CNN alongside the Random Forest & BiLSTM in the model_analysis (Add it to F0 misc)