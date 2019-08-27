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
struct NeuralNetwork{
	Net_layer H_layer;
	Net_layer O_layer;
};
Matrix matrix_sigmoid(Matrix Data); // 所有矩陣數值皆過sigmoid函數
Matrix matrix_sigmoid_der(Matrix Data); // 所有矩陣數值皆過sigmoid導函數
Matrix matrix_loss_function(Matrix Matrix_1,Matrix Matrix_2); // 損失函數( 目標矩陣，輸出矩陣 )
Matrix matrix_loss_function_der(Matrix Matrix_1,Matrix Matrix_2);// 損失導函數( 目標矩陣，輸出矩陣 )
Matrix label_processing(Matrix Data); 
Matrix data_processing(Matrix Data);
Matrix matrix_hidden_layer_error(Matrix weight,Matrix error);
Net_layer create_net_layer(int data_num,int col,int net_num); // 建造神經層，輸入總資料橫行個數，直行個數，神經元個數  
Matrix one_hot_encoding(Matrix data,Matrix label); // 做標籤 
NeuralNetwork net_forward(NeuralNetwork NN,Matrix Data); // 前向傳播 
NeuralNetwork net_back(NeuralNetwork NN,Matrix Label); //倒傳遞 
NeuralNetwork net_update_weight(NeuralNetwork NN,double learning_rate,Matrix Data);
NeuralNetwork net_update_bais(NeuralNetwork NN,double learning_rate);
void printALLData(NeuralNetwork NN);
#endif
