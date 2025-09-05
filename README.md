<div align="center">    

# SSIR: Secure SAGSIN IAB Relay Network

![GitHub release (latest SemVer)](https://img.shields.io/badge/release-v0.1.0-blue)
[![Read the Docs](https://img.shields.io/readthedocs/torch-influence)](asdf/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE.txt)

</div>

SSIR is a comprehensive implementation of multi-hop relaying space-air-ground-sea integrated networks; and an official implementation of the paper titled "-".
This repository contains a variety of routing algorithms including A*(or Dijkstra), genetic algorithm, graph convolutional network (GCN), and graph attention network (GAT).

## Installation

Pip from source:

```bash
git clone https://github.com/hslyu/SSIR
 
cd SSIR
pip install .   
 ```
______________________________________________________________________

## Quickstart
```
from ssir import environment as env

dm = env.DataManager()
pm = env.PlotManager()
master_graph = dm.generate_master_graph()

costs, predecessors_distance = pf.a_star(master_graph, metric="distance")
graph = pf.get_solution_graph(graph, predecessors_distance)

pm.plot_dm(dm, graph)
```
<img src="./example.png" style="width: 600px; height: auto;" title="Code result"/>

______________________________________________________________________

### Overview
______________________________________________________________________

## Contributors
- [Hyeonsu Lyu](https://www.lyu.kr/)
______________________________________________________________________

## Acknowledgements

______________________________________________________________________

## Reference
If you find the code is helpful, please refer our paper!
```
@misc{lyu2025-SSIR,
    title={Secure Multi-Hop Relaying in Large-Scale Space-Air-Ground-Sea Integrated Networks}, 
    author={Hyeonsu Lyu and Hyeonho Noh and Hyun Jong Yang and Kaushik Chowdhury},
    year={2025},
    eprint={2505.00573},
    archivePrefix={arXiv},
    primaryClass={eess.SP},
    url={https://arxiv.org/abs/2505.00573}, 
}
```
