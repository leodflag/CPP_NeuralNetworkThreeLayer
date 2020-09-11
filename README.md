# CPP_NeuralNetworkThreeLayer

CPP_NeuralNetworkThreeLayer
潛在錯誤：
ThreeLayerNNApi.cpp
1. 289、299行 建立隱藏層輸出層時bias要在隱藏層而非輸出層

2. matrix_hidden_layer_error()
76 行 要算隱藏層錯誤時只算到1顆神經元

3. SGD_testing()
資料順序要打亂，或者亂數不重複取值後調資料
資料全部跑完一次後為一次迭代

4. SGD()
循環次數不對

估測錯誤
