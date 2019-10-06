#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h> 
#include <math.h>
#include "ThreeLayerNNApi.hpp"
#include "MatrixOpApi.hpp"
using namespace std;
Matrix matrix_sigmoid(Matrix Data){ // 所有矩陣數值皆過sigmoid函數
	Matrix Data_sig=create_new_matrix(Data.data_row,Data.data_col);
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){
			Data_sig.data_matrix[r][c]=1.0/(1.0+exp(-Data.data_matrix[r][c]));
		}
	}
	return Data_sig;
}
Matrix matrix_sigmoid_der(Matrix Data){ // 所有矩陣數值皆過sigmoid導函數
	Matrix Data_sig_der=create_new_matrix(Data.data_row,Data.data_col);
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){
			Data_sig_der.data_matrix[r][c]=Data.data_matrix[r][c]*(1.0-Data.data_matrix[r][c]);
		}
	}
	return Data_sig_der;
}
Matrix matrix_loss_function(Matrix Matrix_tag,Matrix Matrix_out){ // 損失函數( 目標矩陣，輸出矩陣 )
	Matrix OUT=create_new_matrix(Matrix_out.data_row,Matrix_out.data_col);
	double ans;
	for(int r=0;r<Matrix_tag.data_row;r++){
		for(int c=0;c<Matrix_tag.data_col;c++){
			ans=0.0;
			ans=(Matrix_tag.data_matrix[r][c]-Matrix_out.data_matrix[r][c]);
			OUT.data_matrix[r][c]=(1.0/2.0)*pow(ans,2.0);
		}
	}
	return OUT;
}
Matrix matrix_loss_function_der(Matrix Matrix_tag,Matrix Matrix_out){ // 損失導函數( 目標矩陣，輸出矩陣 )
	Matrix OUT=create_new_matrix(Matrix_out.data_row,Matrix_out.data_col);
	for(int r=0;r<Matrix_tag.data_row;r++){
		for(int c=0;c<Matrix_tag.data_col;c++){
			OUT.data_matrix[r][c]=Matrix_tag.data_matrix[r][c]-Matrix_out.data_matrix[r][c];
		}
	}
	return OUT;
}
Matrix label_processing(Matrix Data){ // label 處理
	Matrix Matrix_D=matrix_equal(Data); // 為避免動到原Data的數值，因此建立一個新空間去存Data的數據 
	Matrix Label=matrix_get_col_label_data(Matrix_D,Data.data_col-1); // 取出類別標籤
	Matrix Label_1=one_hot_encoding(Matrix_D,Label); // one hot encoding方式重新標籤
	return Label_1;
}
Matrix data_processing(Matrix Data){ // data 處理
	Data=matrix_delete_last_col_data(Data); // 切掉類別那行
	Data=matrix_add_col_one(Data); //  填入1，作為bais相乘時的權重
	return Data;
}
Net_layer create_net_layer(int data_num,int col,int net_num){ // 建造神經層，輸入總資料橫行個數，直行個數，神經元個數  
	Net_layer NetLayer;
	NetLayer.w=create_rand_matrix(net_num,col);  // 建立隱藏層的初始權重矩陣  
	NetLayer.delta_w=create_zero_matrix(net_num,col);  // 建立修正權重矩陣 
	NetLayer.net=create_zero_matrix(data_num,net_num);  // 加總(權重*數值)+bais
	NetLayer.net_sigmoid=create_zero_matrix(data_num,net_num);  // 過sigmoid 
	NetLayer.error=create_zero_matrix(data_num,net_num); // layer的error 
	return NetLayer;
}
Matrix matrix_hidden_layer_error(Matrix weight,Matrix error){ // 倒傳遞時計算隱藏層錯誤的function
	Matrix Matrix_sum=create_new_matrix(error.data_row,error.data_col);
	for(int c=0;c<Matrix_sum.data_col;c++){  //Herr_1=Oerr_1*O_1w+Oerr_1*O_2w+Oerr_1*O_3w
		for(int k=0;k<weight.data_col-1;k++){ 
			Matrix_sum.data_matrix[0][c]+=error.data_matrix[0][k]*weight.data_matrix[k][c];
		}
	}
	return Matrix_sum;
}
Matrix one_hot_encoding(Matrix data,Matrix label){  // 給原本的data,抓出的label 
	Matrix label_1=matrix_row_sort_small_to_large(label,0);// 將標籤{1,0}順序改成{0,1) 	
	Matrix goal_matrix=create_new_matrix(data.data_row,label_1.data_col);
	for(int i=0;i<data.data_row;i++){
		for(int j=0;j<goal_matrix.data_col;j++){ // 假設標籤是012，data的標籤是1，轉換結果是010 
			if(data.data_matrix[i][data.data_col-1]==label_1.data_matrix[0][j])
				goal_matrix.data_matrix[i][j]=1.0; // 標籤對應到時就標1 
			else
				goal_matrix.data_matrix[i][j]=0.0; // 沒對應到就標0 
		}
	}
	return goal_matrix;
}
NeuralNetwork net_forward(NeuralNetwork NN,Matrix Data){ // 前向傳播 
	NN.H_layer.w=matrix_tran_last_col_negative(NN.H_layer.w); // 將bais轉成負號 
	NN.H_layer.net=matrix_mult(Data,NN.H_layer.w); // 矩陣乘法，有將權重轉置 
	NN.H_layer.net_sigmoid=matrix_sigmoid(NN.H_layer.net); // hidden_net
	Matrix D=matrix_add_col_one(NN.H_layer.net_sigmoid); // 加了bais 
	NN.O_layer.w=matrix_tran_last_col_negative(NN.O_layer.w); // bais轉成負號
	NN.O_layer.net=matrix_mult(D,NN.O_layer.w); // 矩陣乘法，有將權重轉置，算出Output_net
	NN.O_layer.net_sigmoid=matrix_sigmoid(NN.O_layer.net); // 預測值 
	return NN;	
} 
NeuralNetwork net_back(NeuralNetwork NN,Matrix Label){   // 倒傳遞 
	NN.O_layer.error=matrix_loss_function_der(Label,NN.O_layer.net_sigmoid); // 從輸出層往回推 (T-Y) 期望與輸出間的誤差
	Matrix Sigmoid_der=matrix_sigmoid_der(NN.O_layer.net_sigmoid);  // Sigmoid 導函數 Y(1-Y)  輸出層預測的sigmoid導函數
	NN.O_layer.error=matrix_hadamard(NN.O_layer.error,Sigmoid_der); //使用哈達瑪積矩陣乘法 (T-Y)Y(1-Y) 輸出層誤差
	Matrix Matrix_H_err=matrix_hidden_layer_error(NN.O_layer.w,NN.O_layer.error); // 權重的錯誤相加
	NN.H_layer.error=matrix_sigmoid_der(NN.H_layer.net_sigmoid); //  隱藏層預測的sigmoid導函數
	NN.H_layer.error=matrix_hadamard(NN.H_layer.error,Matrix_H_err); //  隱藏層誤差
    return NN;
}
NeuralNetwork net_update_weight(NeuralNetwork NN,double learning_rate,Matrix Data){ // 更新權重 
	for(int r=0;r<NN.O_layer.delta_w.data_row;r++){
		for(int c=0;c<NN.O_layer.delta_w.data_col-1;c++){ // R*Oerr_1*H_sig 
			NN.O_layer.delta_w.data_matrix[r][c]=learning_rate*NN.O_layer.error.data_matrix[0][r]*NN.H_layer.net_sigmoid.data_matrix[0][c];
		}
	}
	for(int r=0;r<NN.H_layer.delta_w.data_row;r++){
		for(int c=0;c<NN.H_layer.delta_w.data_col-1;c++){
			NN.H_layer.delta_w.data_matrix[r][c]=learning_rate*NN.H_layer.error.data_matrix[0][r]*Data.data_matrix[0][c];
		}
	}
	NN.O_layer.w=matrix_plus(NN.O_layer.delta_w,NN.O_layer.w);
	NN.H_layer.w=matrix_plus(NN.H_layer.delta_w,NN.H_layer.w);
	return NN;
}
NeuralNetwork net_update_bais(NeuralNetwork NN,double learning_rate){ // 更新bais 
	NN.H_layer.w=matrix_tran_last_col_negative(NN.H_layer.w); // 將原本的+bais 在更新時轉回 -bais 
	NN.O_layer.w=matrix_tran_last_col_negative(NN.O_layer.w); // 將原本的+bais 在更新時轉回 -bais  
	for(int r=0;r<NN.O_layer.delta_w.data_row;r++){  
		NN.O_layer.delta_w.data_matrix[r][NN.O_layer.delta_w.data_col-1]=-learning_rate*NN.O_layer.error.data_matrix[0][r];
	}
	for(int r=0;r<NN.H_layer.delta_w.data_row;r++){
		NN.H_layer.delta_w.data_matrix[r][NN.H_layer.delta_w.data_col-1]=-learning_rate*NN.H_layer.error.data_matrix[0][r];
	}
	NN.O_layer.w=matrix_plus(NN.O_layer.delta_w,NN.O_layer.w); // 誤差權重加上權重 
	NN.H_layer.w=matrix_plus(NN.H_layer.delta_w,NN.H_layer.w);
	NN.H_layer.delta_w=re_zero(NN.H_layer.delta_w); // 將權重以外的都歸零 
	NN.O_layer.delta_w=re_zero(NN.O_layer.delta_w);
	NN.H_layer.net=re_zero(NN.H_layer.net);
	NN.O_layer.net=re_zero(NN.O_layer.net);
	NN.H_layer.net_sigmoid=re_zero(NN.H_layer.net_sigmoid);
	NN.O_layer.net_sigmoid=re_zero(NN.O_layer.net_sigmoid);
	NN.H_layer.error=re_zero(NN.H_layer.error);
	NN.O_layer.error=re_zero(NN.O_layer.error);
	return NN;	
}
NeuralNetwork BGD_calculate_delta_weight(NeuralNetwork NN,double learning_rate,Matrix Data){ //計算每筆數值的權重與bais，並全部加起來 
	NN.H_layer.w=matrix_tran_last_col_negative(NN.H_layer.w); // 將原本的+bais(T-O的關係) 在更新時轉回 -bais 
	NN.O_layer.w=matrix_tran_last_col_negative(NN.O_layer.w); // 將原本的+bais 在更新時轉回 -bais 
	for(int r=0;r<NN.O_layer.delta_w.data_row;r++){
		for(int c=0;c<NN.O_layer.delta_w.data_col-1;c++) // delta_w=R*Oerr_1*H_sig // delta_bais=-R*Oerr_1 
			NN.O_layer.delta_w.data_matrix[r][c]+=learning_rate*NN.O_layer.error.data_matrix[0][r]*NN.H_layer.net_sigmoid.data_matrix[0][c];
		NN.O_layer.delta_w.data_matrix[r][NN.O_layer.delta_w.data_col-1]+=-learning_rate*NN.O_layer.error.data_matrix[0][r];
	}
	for(int r=0;r<NN.H_layer.delta_w.data_row;r++){
		for(int c=0;c<NN.H_layer.delta_w.data_col-1;c++)
			NN.H_layer.delta_w.data_matrix[r][c]+=learning_rate*NN.H_layer.error.data_matrix[0][r]*Data.data_matrix[0][c];
		NN.H_layer.delta_w.data_matrix[r][NN.H_layer.delta_w.data_col-1]+=-learning_rate*NN.H_layer.error.data_matrix[0][r];
	}
//	printData(NN.O_layer.delta_w);
//	printData(NN.H_layer.delta_w);
	NN.H_layer.net=re_zero(NN.H_layer.net);
	NN.O_layer.net=re_zero(NN.O_layer.net);
	NN.H_layer.net_sigmoid=re_zero(NN.H_layer.net_sigmoid);
	NN.O_layer.net_sigmoid=re_zero(NN.O_layer.net_sigmoid);
	NN.H_layer.error=re_zero(NN.H_layer.error);
	NN.O_layer.error=re_zero(NN.O_layer.error);
	return NN;
} 
NeuralNetwork BGD_update_weight_and_bais(NeuralNetwork NN,int data_row){ // 更新權重與bais，要將加總的錯誤平均值算出來，
//	double m=0.0;
//	m=1.0/data_row;
//	printf("m=%f \n",m);
//	NN.O_layer.delta_w=matrix_mult_num(NN.O_layer.delta_w,m); // 除以data數 
//	printData(NN.O_layer.delta_w);
//	printData(NN.H_layer.delta_w);
	NN.O_layer.w=matrix_plus(NN.O_layer.delta_w,NN.O_layer.w); // 誤差權重加上權重 
	NN.H_layer.w=matrix_plus(NN.H_layer.delta_w,NN.H_layer.w);
	NN.H_layer.delta_w=re_zero(NN.H_layer.delta_w); // 將權重以外的都歸零 
	NN.O_layer.delta_w=re_zero(NN.O_layer.delta_w);	
	return NN;
} 
void save_weight(NeuralNetwork NN){ // 儲存隱藏層、輸出層權重 
	ofstream writeFile;// 寫文件，會覆蓋掉原本的資訊 
	writeFile.open("hidden_weight.csv",ios::out);
	for(int r=0;r<NN.H_layer.w.data_row;r++){
		for(int c=0;c<NN.H_layer.w.data_col;c++){
			writeFile<<NN.H_layer.w.data_matrix[r][c]<<",";
		}
		writeFile<<endl;
	}
	writeFile.close();
	ofstream writeFile1;
	writeFile1.open("output_weight.csv",ios::out);
	for(int r=0;r<NN.O_layer.w.data_row;r++){
		for(int c=0;c<NN.O_layer.w.data_col;c++){
			writeFile1<<NN.O_layer.w.data_matrix[r][c]<<",";
		}
		writeFile1<<endl;
	}
	writeFile1.close();	
} 
void save_nn_structure(NeuralNetwork NN,int hidden_net_num,int output_net_num,int data_col,double learning_rate,int iteration){ // 儲存神經網路架構 
	ofstream writeFile;// 寫文件，會覆蓋掉原本的資訊 
	writeFile.open("nn_structure.csv",ios::out);
	writeFile<<"input_net_num="<<data_col-1<<endl;
	writeFile<<"hidden_net_num="<<hidden_net_num<<endl;
	writeFile<<"output_net_num="<<output_net_num<<endl;
	writeFile<<"learning_rate="<<learning_rate<<endl;
	writeFile<<"hidden_layer_weight："<<endl;
	for(int r=0;r<NN.H_layer.w.data_row;r++){
		for(int c=0;c<NN.H_layer.w.data_col;c++){
			writeFile<<NN.H_layer.w.data_matrix[r][c]<<",";
		}
		writeFile<<endl;
	}
	writeFile<<"output_layer_weight："<<endl;
	for(int r=0;r<NN.O_layer.w.data_row;r++){
		for(int c=0;c<NN.O_layer.w.data_col;c++){
			writeFile<<NN.O_layer.w.data_matrix[r][c]<<",";
		}
		writeFile<<endl;
	}
	writeFile.close();
} 
void printALLData(NeuralNetwork NN){
	printData(NN.H_layer.delta_w);
	printData(NN.H_layer.error);
	printData(NN.H_layer.net);
	printData(NN.H_layer.net_sigmoid);
	printData(NN.H_layer.w);
	printf("------H-------\n");
	printData(NN.O_layer.delta_w);
	printData(NN.O_layer.error);
	printData(NN.O_layer.net);
	printData(NN.O_layer.net_sigmoid);
	printData(NN.O_layer.w);
	printf("------O-------\n");
}
void SGD(Matrix Data,int hidden_net_num,int output_net_num,int data_col,double learning_rate,int iteration,double stop_err){
	int data_order=0,OK_data=0,cout_err_num=0,forward_label=0,real_label=0;
	double err_num,err_rate=0.0;
	NeuralNetwork NN;
	NN.H_layer=create_net_layer(1,data_col,hidden_net_num); // 建立隱藏層 
	NN.O_layer=create_net_layer(1,hidden_net_num+1,output_net_num); // 輸出層和隱藏層矩陣運算時，col要加上bais 
	Matrix Label=label_processing(Data); // label 處理 
	printData(Label);
	Data=data_processing(Data); // data 處理 
	while(iteration>0){ // 循環次數 
		while(data_order<Data.data_row){ // 循環一筆筆資料 
			Matrix DATA=matrix_get_one_row_data(Data,data_order); // 取得一筆資料 
			NN=net_forward(NN,DATA); // 前向傳播 
			Matrix Label_1=matrix_get_one_row_data(Label,data_order); // 取得同列的label
			// 檢查預測標籤對不對 
			forward_label=matrix_find_max_col(NN.O_layer.net_sigmoid);
			real_label=matrix_find_max_col(Label_1);
			if(forward_label!=real_label)
				cout_err_num++;	
			if(iteration==1) // 只印出最後一筆的預測結果 
				printData(NN.O_layer.net_sigmoid);
			err_num=0.0;
			Matrix ERROR=matrix_loss_function(Label_1,NN.O_layer.net_sigmoid); // 計算error
			err_num=matrix_total(ERROR);
			if(err_num<stop_err) // 誤差值停止條件 
				OK_data++;
//			if(iteration==1) //如果提早結束就不會出現了 
//				printData(ERROR);
			NN=net_back(NN,Label_1); // 倒傳遞 
			NN=net_update_weight(NN,learning_rate,DATA); // 計算並更新權重 
			NN=net_update_bais(NN,learning_rate);  // 計算並更新bais 
			data_order++;		
		}
//		printf("------iteration=%d------\n",iteration);
		if(OK_data==Data.data_row) // 如果分類結果小於指定誤差的行數已經是全部了，就停 
			break;
		else{
			err_rate=(double)cout_err_num/Data.data_row;
			if(iteration==1)
				printf("err_rate=%f\n",err_rate);
			cout_err_num=0;
			OK_data=0;   // 分類訓練OK的數量歸零 
			data_order=0; // 從頭訓練 
			iteration--;  // 迭代次數減 1  
		}
	}
	save_nn_structure(NN,hidden_net_num,output_net_num,data_col,learning_rate,iteration);
	double number;
	Matrix testData=create_one_matrix(1,data_col);
	string ans="y",str;
	while(ans=="y"){
		int feature=data_col-1;
		int c=0;
		while(feature!=0){
			printf("請輸入測試資料：");
			while( !( cin >> number ) ){
				cin.clear(); //清除 ios_base::failbit
				getline( cin, str ); //清掉一行
				cout << "請輸入數字" << endl;
		  	}
			testData.data_matrix[0][c]=number;
			cout<<testData.data_matrix[0][c]<<endl;
			c++;
			feature--;
		}
		printData(NN.H_layer.w);
		NN=net_forward(NN,testData); // 前向傳播 
		printf("預測結果為：\n");
		printData(NN.O_layer.net_sigmoid);	
		printf("是否繼續預測? y/n \n");
		cin>>ans;	
		NN.H_layer.w=matrix_tran_last_col_negative(NN.H_layer.w); // bais轉成負號
		NN.O_layer.w=matrix_tran_last_col_negative(NN.O_layer.w); // bais轉成負號
	}
}
void BGD(Matrix Data,int hidden_net_num,int output_net_num,int data_col,double learning_rate,int iteration,double stop_err){ // 隨機梯度下降 
	int data_order=0,OK_data=0;
	double err_num;
	NeuralNetwork NN;
	NN.H_layer=create_net_layer(1,data_col,hidden_net_num); // 建立隱藏層 
	NN.O_layer=create_net_layer(1,hidden_net_num+1,output_net_num); // 輸出層和隱藏層矩陣運算時，col要加上bais 
	Matrix Label=label_processing(Data); // label 處理 
	Data=data_processing(Data); // data 處理 
	while(iteration>0){ // 循環次數 
		while(data_order<Data.data_row){ // 循環一筆筆資料 
			Matrix DATA=matrix_get_one_row_data(Data,data_order); // 取得一筆資料 
			NN=net_forward(NN,DATA); // 前向傳播 
//			printData(DATA);
//			printData(NN.O_layer.net_sigmoid)
			Matrix Label_1=matrix_get_one_row_data(Label,data_order); // 取得同列的label 
//			printData(Label_1);
			if(iteration==1) // 只印出最後一筆的預測結果 
				printData(NN.O_layer.net_sigmoid);
			err_num=0.0;	
			Matrix ERROR=matrix_loss_function(Label_1,NN.O_layer.net_sigmoid); // 計算error
			err_num=matrix_total(ERROR);
			if(err_num<stop_err*2)
				OK_data++;
//			if(iteration==1)
//				printData(ERROR);
			NN=net_back(NN,Label_1); // 倒傳遞 
//			printALLData(NN);
//			printf("===back----------back===\n");
			NN=BGD_calculate_delta_weight(NN,learning_rate,DATA); //計算每筆數值的權重與bais並加起來 
//			printALLData(NN);
//			printf("------data_order=%d------\n\n",data_order);
			data_order++;		
		}
		if(OK_data==Data.data_row)
			break;
		else{
			NN=BGD_update_weight_and_bais(NN,4);
			OK_data=0;
			data_order=0;
			iteration--;			
		}
//		printf("------iteration=%d------\n",iteration);
	}
	save_nn_structure(NN,hidden_net_num,output_net_num,data_col,learning_rate,iteration);
	double number;
	Matrix testData=create_one_matrix(1,data_col);
	string ans="y",str;
	while(ans=="y"){
		int feature=data_col-1;
		int c=0;
		while(feature!=0){
			printf("請輸入測試資料：");
			while( !( cin >> number ) ){
				cin.clear(); //清除 ios_base::failbit
				getline( cin, str ); //清掉一行
				cout << "請輸入數字" << endl;
		  	}
//			if(number)
			testData.data_matrix[0][c]=number;
			cout<<testData.data_matrix[0][c]<<endl;
			printData(testData);
			c++;
			feature--;
			
		}
		printData(testData);
		NN=net_forward(NN,testData); // 前向傳播 
		printf("預測結果為：\n");
		printData(NN.O_layer.net_sigmoid);	
		printf("是否繼續預測? y/n \n");
		cin>>ans;	
		NN.H_layer.w=matrix_tran_last_col_negative(NN.H_layer.w); // bais轉成負號
		NN.O_layer.w=matrix_tran_last_col_negative(NN.O_layer.w); // bais轉成負號
	}
} 
