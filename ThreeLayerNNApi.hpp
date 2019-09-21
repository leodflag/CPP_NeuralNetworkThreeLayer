#ifndef THREELAYERNNAPI_HPP
#define THREELAYERNNAPI_HPP
#include "MatrixOpApi.hpp"
struct Net_layer{ // 類神經網路層，矩陣形式 
	Matrix w;  // 權重矩陣，含bais 
	Matrix delta_w; // 權重誤差 
	Matrix net; // 輸入*權重總和 row=>資料個數  col=>神經元個數
	Matrix net_sigmoid; // 預測值  row=>資料個數  col=>神經元個數
	Matrix error; // 誤差值矩陣  row=>資料個數  col=>神經元個數
};
struct NeuralNetwork{  // 三層神經網路的結構 
	Net_layer H_layer; // 隱藏層 
	Net_layer O_layer; // 輸出層 
};
Matrix matrix_sigmoid(Matrix Data); // 所有矩陣數值皆過sigmoid函數
Matrix matrix_sigmoid_der(Matrix Data); // 所有矩陣數值皆過sigmoid導函數
Matrix matrix_loss_function(Matrix Matrix_1,Matrix Matrix_2); // 損失函數( 目標矩陣，輸出矩陣 )
Matrix matrix_loss_function_der(Matrix Matrix_1,Matrix Matrix_2);// 損失導函數( 目標矩陣，輸出矩陣 )
Matrix label_processing(Matrix Data);  // label 處理，取出類別標籤後用one hot encoding方式重新標籤 
Matrix data_processing(Matrix Data);   // data 處理，切掉類別那行，填入1，作為bais相乘時的權重 
Matrix matrix_hidden_layer_error(Matrix weight,Matrix error); // 倒傳遞時計算隱藏層錯誤的function 
Net_layer create_net_layer(int data_num,int col,int net_num); // 建造神經層，輸入總資料橫行個數，直行個數，神經元個數  
Matrix one_hot_encoding(Matrix data,Matrix label); // 標籤方式，有幾類，輸出層就有幾個神經元 
NeuralNetwork net_forward(NeuralNetwork NN,Matrix Data); // 前向傳播 
NeuralNetwork net_back(NeuralNetwork NN,Matrix Label); //倒傳遞 
NeuralNetwork net_update_weight(NeuralNetwork NN,double learning_rate,Matrix Data); // 計算並更新權重 
NeuralNetwork net_update_bais(NeuralNetwork NN,double learning_rate); // 計算並更新bais 
NeuralNetwork BGD_calculate_delta_weight(NeuralNetwork NN,double learning_rate,Matrix Data); //計算每筆數值的權重與bais並加起來
NeuralNetwork BGD_update_weight_and_bais(NeuralNetwork NN,int data_row); // 更新權重與bais，要將加總的錯誤平均值算出來，除以data數 
void save_weight(NeuralNetwork NN); // 儲存隱藏層、輸出層權重 
void save_nn_structure(NeuralNetwork NN,int hidden_net_num,int output_net_num,int data_col,double learning_rate,int iteration); // 儲存神經網路架構 
void printALLData(NeuralNetwork NN); // 印出神經網路隱藏層、輸出層的數值 
void SGD(Matrix Data,int hidden_net_num,int output_net_num,int data_col,double learning_rate,int iteration,double stop_err); // 隨機梯度下降 
void BGD(Matrix Data,int hidden_net_num,int output_net_num,int data_col,double learning_rate,int iteration); // 隨機梯度下降 
#endif
