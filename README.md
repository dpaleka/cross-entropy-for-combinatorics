# cross-entropy-for-combinatorics
Code accompanying the manuscript ["Constructions in combinatorics via neural networks" by A Z
Wagner](https://arxiv.org/abs/2104.14516).

The following code allows you to optimize any score function on the space of graphs with
a fixed number of vertices.

## Demo
![On the left we see a terminal with the function values and adjacency values. 
On the right we see the best graph in each step.](demo.gif)
The algorithm learns to generate graphs with low proximity + distance eigenvalue, 
as in Conjecture 2.3. in the paper.

## Installation instructions
Clone this repository as follows:
```
git clone https://github.com/dpaleka/cross-entropy-for-combinatorics
cd cross-entropy-for-combinatorics
```

Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
on your system, and run
```
conda env create --file environment.yml 
conda activate cross-entropy-for-combinatorics
```

## How to use this for a custom score function?
Replace the `score_graph` method in `score.py` with the function you want to **maximize**.

For example, to minimize the absolute ratio of the first and the last eigenvalue of the adjacency matrix of a connected graph, run:
```
def score_graph(adjMatG, edgeListG, Gdeg):
    """
    Reward function. The arguments adjMatG, edgeListG, Gdeg are numpy arrays.
    """
    N = Gdeg.size
    INF = 100000

    _, conn = bfs(Gdeg,edgeListG)
    if not conn:
        return -INF
        
    lambdas = np.flip(np.sort(np.linalg.eigvals(adjMatG)))
    return -abs(lambdas[0]/lambdas[-1])
```

Check the parameters at the top of `optimize.py`, in particular the number of vertices `N`. 
Then run
```
python optimize.py
```

Of course, this will just generate connected bipartite graphs after a few iterations.


## Improving performance
If the generated graphs are not improving with regards to your score function,
there are some straightforward directions to try:

1. Modify the parameters at the top of `optimize.py`. In particular, increasing the `super_percentile` parameter
   forces more exploration, and decreasing the `LEARNING_RATE` parameter can get you out of local optima.
   
3. Use a more powerful model to parametrize the graph generation.
   See [You et al., 2018](https://arxiv.org/abs/1802.08773) 
   (it has [code for the Graph RNN](https://github.com/JiaxuanYou/graph-generation))
   or the permutation-equivariant [Li et al., 2018](https://arxiv.org/abs/1803.03324).

3. Change the loss function. There is a variety of [policy
   gradient algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)
   to choose from.

