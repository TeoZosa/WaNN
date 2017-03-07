#Breakthrough CNN
##### Deep neural networks for use in an agent utilizing a Monte Carlo tree search in the game of Breakthrough. 
<div align="center">
<a href="http://www.trmph.com/breakthrough/board#8,g2h3b7b6h1g2c7c6h3g4g7f6h2g3h8g7g3f4h7g6b2c3b6c5a1b2a8b7a2b3a7b6e2d3g6g5d2e3d7e6b3c4b6b5f2g3e7d6g2f3d8e7c2b3d6e5b3b4e7d6f3e4b7b6d1c2e8d7e1f2d6d5c4d5c6d5f2f3d7d6d3d4g7g6c2d3c5c4c1c2e5d4e3d4e6e5f4e5f6e5g1f2f7e6f2e3e5d4e3d4e6e5f1f2c4d3c2d3d5e4d3e4b5c4b1c2e5d4c3d4b6b5g4f5c4c3b2c3b5a4f5g6a4a3d4d5a3b2d5c6b2c1"><img src="https://cloud.githubusercontent.com/assets/13070236/23641302/872ea850-02a7-11e7-9be4-49c37f803c27.JPG" title="Breakthrough game board" style="width:302px;height:348px;"></img></a>
  <img src="https://cloud.githubusercontent.com/assets/13070236/23594196/f8bf7854-01cc-11e7-9823-4e0a9bd4d2b8.png" title="One of the feature planes as seen by the neural network"></img>
</div>
-----------------
- To see an example of a full game between the policy net and an agent using the policy net for node expansion in MCTS, click the game board image above (courtesy of www.trmph.com)
- The neural networks herein will eventually be merged with a C++ program that uses a Monte Carlo tree search.<p>
- Through [`breakthrough_player.py`](../master/Breakthrough_Player/breakthrough_player.py), a user can play against a trained policy net, or MCTS+policy net variants.<p> 
- A majority of the code is made up of multiple auxiliary files for pulling the data from a <a href="https://www.littlegolem.net/jsp/games/gamedetail.jsp?gtid=brkthr">website storing Breakthrough game data</a>, or self-play logs, and transforming that data into labeled training data for the neural networks (built using <a href="https://github.com/tensorflow/tensorflow"> Tensorflow</a>).<p><p> 
- Currently running experiments on different models trained on self-play data.


