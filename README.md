#Breakthrough ANN
##### Deep neural networks for use in an agent utilizing a Monte Carlo tree search in the game of Breakthrough. 
<div align="center">
  <img src="https://cloud.githubusercontent.com/assets/13070236/22083285/2f75ff60-dd80-11e6-821d-47d3e41cc9a9.png" title="One of the feature planes as seen by the neural network">
</div>
-----------------
- The neural networks herein will eventually be merged with a C++ program that uses a Monte Carlo tree search.<p>
- Through [`breakthrough_player.py`](../master/Breakthrough_Player/breakthrough_player.py), a user can play against a trained policy net, or MCTS+policy net variants.<p> 
- A majority of the code is made up of multiple auxiliary files for pulling the data from a <a href="https://www.littlegolem.net/jsp/games/gamedetail.jsp?gtid=brkthr">website storing Breakthrough game data</a>, or self-play logs, and transforming that data into labeled training data for a neural network built using <a href="https://github.com/tensorflow/tensorflow"> Tensorflow</a>.<p><p> 
- Currently running experiments on different models trained on self-play data.
