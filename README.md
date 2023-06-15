# DeepMOIS-MC (Deep Multi-Omics Integrative Subtyping by Maximizing Correlation)


**Paper Name:-** [Deep Multi-Omics Integration by Learning Correlation-Maximizing Representation Identifies Prognostically Stratified Cancer Subtypes]()
* **Authors:** Yanrong Ji<sup>1</sup>, Pratik Dutta<sup>2</sup>, and Ramana Davuluri<sup>2</sup>
* **Affiliation:** <sup>1</sup>1Driskill Graduate Program in Life Sciences, Northwestern University, Chicago, IL, <sup>2</sup>2Department of Biomedical Informatics, Stony Brook Cancer Center, Stony Brook Medicine, Stony Brook University, Stony Brook, NY 11794,
* **Accepted(2nd June, 2023):** [Bioinformatics Advances| Oxford Academic](https://academic.oup.com/bioinformaticsadvances)
* **Corresponding Author:** [Ramana V. Davuluri](https://bmi.stonybrookmedicine.edu/people/ramana_davuluri) (Ramana.Davuluri@stonybrookmedicine.edu ) 

## Acknowledgement
Note that we didn't start everything from scratch but modified the source code from [jameschapman19/cca_zoo](https://github.com/jameschapman19/cca_zoo). We are very thankful to James Chapman and Hao-Ting Wang - authors of this original version. 


**NOTE**: We have modified the codebase for generating the shared representation of the multi-omics data. In this work, we have used three omics views, e.g. Methylation, RNASeq and miRNA. But the code is written in a way so that it will works for any number of views with passing diffrent arguments.  

## Abstract
We developed a novel outcome-guided molecular subgrouping framework, called DeepMOIS-MC (Deep Multi-Omics Integrative Subtyping by Maximizing Correlation), for integrative learning from multi-omics data by maximizing correlation between all input -omics views. DeepMOIS-MC consists of two parts: clustering and classification. In the clustering part, the preprocessed high-dimensional multi-omics views are input into two-layer fully connected neural networks. The outputs of individual networks are subjected to Generalized Canonical Correlation Analysis loss to learn the shared representation. Next, the learned representation is filtered by a regression model to select features that are related to a covariate clinical variable, for example, a survival/outcome. The filtered features are used for clustering to determine the optimal cluster assignments. In the classification stage, the original feature matrix of one of the -omics view is scaled and discretized based on equal frequency binning, and then subjected to feature selection using RandomForest. Using these selected features, classification models (for example, XGBoost model) are built to predict the molecular subgroups that were identified at clustering stage. We applied DeepMOIS-MC on lung and liver cancers, using TCGA datasets. In compar-
ative analysis, we found that DeepMOIS-MC outperformed traditional approaches in patient stratification. Finally, we validated the robustness and generalizability of the classification models on independent datasets. We anticipate that the DeepMOIS-MC can be adopted to many multi-omics integrative analyses tasks.


## Usage
* `cd examples/scripts`
* Then run the following bash script
`./dgcca_embedding.sh` 
  - This will generate the shared representation. 
* In the `postprocessing` folder run `UniCoxPh_clustering_KMplot_all_views.ipynb` for generating the clustering results and also generate KM Plot for survival prediction.  
