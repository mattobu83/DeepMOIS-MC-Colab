# DeepMOIS-MC (Deep Multi-Omics Integrative Subtyping by Maximizing Correlation)


**Paper Name:-** [Deep Multi-Omics Integration by Learning Correlation-Maximizing Representation Identifies Prognostically Stratified Cancer Subtypes]()
* **Authors:** Yanrong Ji<sup>1</sup>, Pratik Dutta<sup>2</sup>, and Ramana Davuluri<sup>2</sup>
* **Affiliation:** <sup>1</sup>1Driskill Graduate Program in Life Sciences, Northwestern University, Chicago, IL, <sup>2</sup>2Department of Biomedical Informatics, Stony Brook Cancer Center, Stony Brook Medicine, Stony Brook University, Stony Brook, NY 11794,
* **Accepted(2nd June, 2023):** [Bioinformatics Advances| Oxford Academic](https://academic.oup.com/bioinformaticsadvances)
* **Corresponding Author:** [Ramana V. Davuluri](https://bmi.stonybrookmedicine.edu/people/ramana_davuluri) (Ramana.Davuluri@stonybrookmedicine.edu ) 

## Acknowledgement
Note that we didn't start everything from scratch but modified the source code from [jameschapman19/cca_zoo](https://github.com/jameschapman19/cca_zoo). We are very thankful to James Chapman and Hao-Ting Wang - authors of this original version. 


**We have modified the codebase for generating the shared representation of the multi-omics data. In this work, we have used three omics views, e.g. Methylation, RNASeq and miRNA. But the code is written in a way so that it will works for any number of views with passing diffrent arguments.**  

## Abstract
In this GitHub repository, we have impemented  [DGCCA (Deep Generalized CCA)](https://www.aclweb.org/anthology/W19-4301.pdf) for generating shared representation of multiple views. 



## Usage
* `cd examples/scripts`
* Then run the following bash script
`./dgcca_embedding.sh` 
  - This will generate the shared representation. 
* In the `postprocessing` folder run `UniCoxPh_clustering_KMplot_all_views.ipynb` for generating the clustering results and also generate KM Plot for survival prediction.  
