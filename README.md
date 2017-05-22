
# WaNN
##### Deep neural networks and Monte Carlo tree search in the game of Breakthrough. 
<div align="center">
<a href="http://www.trmph.com/breakthrough/board#8,g2h3b7b6h1g2c7c6h3g4g7f6h2g3h8g7g3f4h7g6b2c3b6c5a1b2a8b7a2b3a7b6e2d3g6g5d2e3d7e6b3c4b6b5f2g3e7d6g2f3d8e7c2b3d6e5b3b4e7d6f3e4b7b6d1c2e8d7e1f2d6d5c4d5c6d5f2f3d7d6d3d4g7g6c2d3c5c4c1c2e5d4e3d4e6e5f4e5f6e5g1f2f7e6f2e3e5d4e3d4e6e5f1f2c4d3c2d3d5e4d3e4b5c4b1c2e5d4c3d4b6b5g4f5c4c3b2c3b5a4f5g6a4a3d4d5a3b2d5c6b2c1"><img src="https://cloud.githubusercontent.com/assets/13070236/23641302/872ea850-02a7-11e7-9be4-49c37f803c27.JPG" title="Breakthrough game board" style="width:302px;height:348px;"></img></a>
  <img src="https://cloud.githubusercontent.com/assets/13070236/23594196/f8bf7854-01cc-11e7-9823-4e0a9bd4d2b8.png" title="One of the feature planes as seen by the neural network"></img>
</div>

-----------------

- To see an example of a full game between the policy net and an agent using the policy net for node expansion in MCTS, click the game board image above (courtesy of www.trmph.com)

## Python 3.5

This code is written in Python 3.5 and Cython syntax. Click the following links to install the latest version of [Python](https://www.python.org/downloads) and [Cython](http://cython.readthedocs.io/en/latest/src/quickstart/install.html).

## Highlights

- Through [`WaNN_vs_opponent.py`](../master/Breakthrough_Player/WaNN_vs_opponent.py), a user can play against `WaNN` or `WaNN`'s policy net.

- In [`self_play_to_data_structure`](../master/self_play_files/self_play_to_data_structure/), there are multiple auxiliary files (called from [`__main__.py`](../master/main.py)) for parsing the text files generated from self-play games by <a href="http://www.springer.com/cda/content/document/cda_downloaddocument/9783319279916-c2.pdf?SGWID=0-0-45-1545168-p177846880">`Wanderer`</a>, and transforming them into labeled training data for the neural networks (built using <a href="https://github.com/tensorflow/tensorflow"> Tensorflow</a>). 
<br>NOTE: File destinations in the pipeline must be changed manually. In addition, these files use multiprocessing heavily. If you have a large corpus of data and do not have many processors with a significant amount of RAM (in my case, 16 and 128 GiB, respectively), use with caution (unless you want to render your computer unresponsive for a significant length of time).
- The neural network training code resides in [`player_and_opponent_policy_nets.py`](../master/self_play_files/policy_net/player_and_opponent_policy_nets.py) (spaghetti as of 05/20/2017).
- Currently optimizing `WaNN` based on feedback from games against Wanderer.

## Forthcoming
- Command line arguments to [`WaNN_vs_opponent.py`](../master/Breakthrough_Player/WaNN_vs_opponent.py) for a more streamlined interface
- `WaNN` entirely rewritten in C/C++.
- Reinforcement Learning (maybe).

## Deprecated
- In [`little_golem_players_files/tools`](../master/little_golem_players_files/tools), there are multiple auxiliary files for scraping Breakthrough games from <a href="https://www.littlegolem.net/jsp/games/gamedetail.jsp?gtid=brkthr">LittleGolem</a>. 

# Acknowledgements

Many thanks to Dr. Richard Lorentz for allowing the use of Wanderer for this project, as well as his continued support.

