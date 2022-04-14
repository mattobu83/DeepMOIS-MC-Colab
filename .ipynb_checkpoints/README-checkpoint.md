# Multiview_clustering_DGCCA
Implementation of DGCCA which is in form of a python library.

In this project, we are utilizing This implementation can be used to solve multivariate (more than one dimensions) multi-objective optimization problem. The number This implementation can be used to solve multivariate (more than one dimensions) multi-objective optimization problem. The number of objectives and dimensions are not limited. Some critical operators are chosen as: Binary Tournament Selection, Simulated Binary Crossover and Polynomial Mutation. 

Note that we didn't start everything from scratch but modified the source code from [jameschapman19/cca_zoo](https://github.com/jameschapman19/cca_zoo). We are very thankful to James Chapman and Hao-Ting Wang - authors of this original version. The following items are modified:
* Fix the crowding distance formula.
* Modify some parts of the code to apply to any number of objectives and dimensions.
* Modify the selection operator to Tournament Selection.
* Change the crossover operator to Simulated Binary Crossover.
* Change the mutation operator to Polynomial Mutation
