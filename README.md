# Multiview_clustering_DGCCA
In this GitHub repository, we have impemented  [DGCCA (Deep Generalized CCA)](https://www.aclweb.org/anthology/W19-4301.pdf) for generating shared representation of multiple views. The code is written in a way so that it will works for any number of views with passing diffrent arguments.  

Note that we didn't start everything from scratch but modified the source code from [jameschapman19/cca_zoo](https://github.com/jameschapman19/cca_zoo). We are very thankful to James Chapman and Hao-Ting Wang - authors of this original version. The following items are modified:
* Fix the crowding distance formula.
* Modify some parts of the code to apply to any number of objectives and dimensions.
* Modify the selection operator to Tournament Selection.
* Change the crossover operator to Simulated Binary Crossover.
* Change the mutation operator to Polynomial Mutation
