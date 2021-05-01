# cross-entropy-for-combinatorics
Code accompanying the manuscript "Constructions in combinatorics via neural networks" by A Z Wagner
https://arxiv.org/abs/2104.14516

The following code allows you to optimize any score function on the space of graphs with
a fixed number of vertices.

The conj23_pytorch_with_numba.py file contains the solution to Conjecture 2.3. 
It demonstrates the use of numba to speed up the calculation of the reward. 

## Demo
![On the left we see a terminal with the function values and adjacency values. 
On the right we see the best graph in each step.](demo.gif)
The algorithm learns to generate graphs with low proximity + distance eigenvalue, 
as in Conjecture 2.3. in the paper.

## Installation instructions
Clone this repository as follows:
```
git clone https://github.com/zawagner22/cross-entropy-for-combinatorics
cd cross-entropy-for-combinatorics
```

Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
on your system, and run
```
  conda env create --file enviroment.yml 
  conda activate cross-entropy-for-combinatorics
```

## How to use this for a custom score function?
Replace the `score_graph` method in `score.py` with the function you want to *maximize*.
Modify the parameters at the top of `optimize.py`, in particular the number of vertices `N`. 

Then run
```
  python optimize.py
```

## Improving performance
If the generated graphs are not improving with regards to your score function,
there are two straightforward directions to try:

1. Use a more powerful model to parametrize the graph generation.
   See [You et al., 2018](https://arxiv.org/abs/1802.08773) (it has [code]())
   or the permutation-equivariant [Li et al., 2018](https://arxiv.org/abs/1803.03324).

2. Change the loss function. There is a variety of [policy
   gradient algorithsm](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)
   to choose from.

