To setup:
* Create a [new conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
* Activate new conda environment
* Download requirements.txt file (`pip install -r requirements.txt`)
* Download [DELPHI](https://github.com/lucian-ilie/DELPHI) & point its top directory to this one
    * (For the sake of scrapiness actually already have it downloaded so skip this step, we'll move it out of this directory later)

To run:
* `python src/runner.py` 


TODOs:
* Finalize the model.
* Simplify figure 1 ... don't use real data points.
   * BIorender
* (Yitong) Weblogos from the heatmaps as well...
* (Soojung) Investigate what's lurking in parts of the UMAP that we don't think are 12ca5 or MDM2 binders.
    * Are they non-specific binders?
    * look at individual sequences / heatmaps
* (Yehlin) Compare round 2 & round 3 data.
* (Yehlin?) Add the volcano plots with markers of our predicted peptides into F1 notebook
* (Yitong) Generate list of 100 peptides which we most strongly predict for 12ca5 & MDM2 to F3
* Recall / Precision: 10 KFold validation
* Model benchmarking:
    * (Yehlin) Benchmark CNN alongside the Random Forest & BiLSTM in the model_analysis (Add it to F0 misc) (/Users/yitongtseo/Documents/GitHub/ml_phage_display/src/analysis/model_run_comparisons.py)
    * Probably wanna do KFold rather than just one model training per.
    * Futz with matplotlib till all of the matrices are generated onto the same figure automatically.
* Shapely analysis 
    * (Yitong) Create the aggregated figure (grouped by hit motifs for 12ca5 & MDM2)
    * Do some clustering?
    * Add error bars?
* Prettify:
    * Make the hypothetical maximum hit curves match the color of their actual measured hit curves.

* Engineering:
    * (Yehlin) Remove all "X" containing peptides from the dataset (don't know if there even are any left)
        * Then remove the X from the amino acid one hot encoding...

