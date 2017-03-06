#Breakthrough CNN
##### Deep neural networks for use in an agent utilizing a Monte Carlo tree search in the game of Breakthrough. 
<div align="center">
<a href="http://www.trmph.com/breakthrough/board#8,g2h3b7a6h1g2a8b7h3g4a6b5h2g3g7f6b2c3h8g7a1b2h7g6a2b3a7b6g3f4b6c5b3b4e7d6e2d3g6g5d2e3d7e6f2g3c7b6g2f3b7c6e1d2d8e7e3d4e6e5d2e3c6d5d1d2e8d7b2b3d7e6d3e4g7g6c2d3g5f4g3f4g6g5f4e5f6e5g4h5g5h4h5h6h4h3f3f4e7f6f4f5c5c4c1c2c4d3c2d3b8c7c3c4c7c6b1c2c8c7f5e6f7e6c4c5f6g5e4f5e6f5d4e5d6e5c5d6c7d6b4c5b6c5e3d4b5b4f1e2b4a3d4e5h3g2e5d6g2h1"><img src="https://cloud.githubusercontent.com/assets/13070236/23594165/918c8f32-01cc-11e7-9f9e-f826f1991fd0.JPG" title="Breakthrough game board" style="width:302px;height:348px;"></img></a>
  <img src="https://cloud.githubusercontent.com/assets/13070236/23594196/f8bf7854-01cc-11e7-9823-4e0a9bd4d2b8.png" title="One of the feature planes as seen by the neural network"></img>
</div>
-----------------
- To see an example of a full game between the policy net and an agent using the policy net for node expansion in MCTS, click the game board image above (courtesy of www.trmph.com)
- The neural networks herein will eventually be merged with a C++ program that uses a Monte Carlo tree search.<p>
- Through [`breakthrough_player.py`](../master/Breakthrough_Player/breakthrough_player.py), a user can play against a trained policy net, or MCTS+policy net variants.<p> 
- A majority of the code is made up of multiple auxiliary files for pulling the data from a <a href="https://www.littlegolem.net/jsp/games/gamedetail.jsp?gtid=brkthr">website storing Breakthrough game data</a>, or self-play logs, and transforming that data into labeled training data for a neural network built using <a href="https://github.com/tensorflow/tensorflow"> Tensorflow</a>.<p><p> 
- Currently running experiments on different models trained on self-play data.


