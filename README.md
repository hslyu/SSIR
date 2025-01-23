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
dm = env.DataManager()
pm = env.PlotManager()
graph = dm.generate_master_graph()

# 2. Run the A* algorithm
costs, predecessors = pf.a_star(graph, metric="distance")
graph_astar_distance = pf.get_solution_graph(graph, predecessors)

costs, predecessors = pf.a_star(graph, metric="hop")
graph_astar_hop = pf.get_solution_graph(graph, predecessors)

graph_list = [graph_astar_distance, graph_astar_hop]

print(f"A* distance throughput: {graph_astar_distance.compute_network_throughput()}")
print(f"A* hop throughput: {graph_astar_hop.compute_network_throughput()}")
pm.plot_dm(dm, graph_list)
```
<img src="./example.png" title="Code result"/>

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
@ARTICLE{10791413,
  author={Lyu, Hyeonsu and Jang, Jonggyu and Lee, Harim and Yang, Hyun Jong},
  journal={IEEE Transactions on Wireless Communications}, 
  title={Non-iterative Optimization of Trajectory and Radio Resource for Aerial Network}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Trajectory;Optimization;Quality of service;Wireless communication;Downlink;Internet of Things;Resource management;Power control;Iterative methods;Markov decision processes;Trajectory-planning;user association;resource allocation;power control;quality-of-service;Markov decision process},
  doi={10.1109/TWC.2024.3510043}}
```
