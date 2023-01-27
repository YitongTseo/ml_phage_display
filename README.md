To setup:
* Create a [new conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
* Activate new conda environment
* Download requirements.txt file (`pip install -r requirements.txt`)
* Download [DELPHI](https://github.com/lucian-ilie/DELPHI) & point its top directory to this one
    * (For the sake of scrapiness actually already have it downloaded so skip this step, we'll move it out of this directory later)

To run:
* `python src/runner.py` 


TODOs:
* Investigate what's lurking in parts of the UMAP that we don't think are 12ca5 or MDM2 binders.
    * Are they non-specific binders?
* (Yehlin?) Add the volcano plots with markers of our predicted peptides into F1 notebook
* (Yitong) Generate list of 100 peptides which we most strongly predict for 12ca5 & MDM2 to F3
* Model benchmarking:
    * Benchmark CNN alongside the Random Forest & BiLSTM in the model_analysis (Add it to F0 misc)
    * Probably wanna do KFold rather than just one model training per.
    * Futz with matplotlib till all of the matrices are generated onto the same figure automatically.
* Shapely analysis 
    * Create the aggregated figure (grouped by hit motifs for 12ca5 & MDM2)
    * Do some clustering?
    * Add error bars?
* Prettify:
    * Make the hypothetical maximum hit curves match the color of their actual measured hit curves.

* Engineering:
    * Remove all "X" containing peptides from the dataset (don't know if there even are any left)
        * Then remove the X from the amino acid one hot encoding...

