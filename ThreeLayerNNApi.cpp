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
Matrix matrix_sigmoid(Matrix Data){ // �Ҧ��x�}�ƭȬҹLsigmoid���
	Matrix Data_sig=create_new_matrix(Data.data_row,Data.data_col);
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){
			Data_sig.data_matrix[r][c]=1.0/(1.0+exp(-Data.data_matrix[r][c]));
		}
	}
	return Data_sig;
}
Matrix matrix_sigmoid_der(Matrix Data){ // �Ҧ��x�}�ƭȬҹLsigmoid�ɨ��
	Matrix Data_sig_der=create_new_matrix(Data.data_row,Data.data_col);
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){
			Data_sig_der.data_matrix[r][c]=Data.data_matrix[r][c]*(1.0-Data.data_matrix[r][c]);
		}
	}
	return Data_sig_der;
}
Matrix matrix_loss_function(Matrix Matrix_tag,Matrix Matrix_out){ // �l�����( �ؼЯx�}�A��X�x�} )
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
Matrix matrix_loss_function_der(Matrix Matrix_tag,Matrix Matrix_out){ // �l���ɨ��( �ؼЯx�}�A��X�x�} )
	Matrix OUT=create_new_matrix(Matrix_out.data_row,Matrix_out.data_col);
	for(int r=0;r<Matrix_tag.data_row;r++){
		for(int c=0;c<Matrix_tag.data_col;c++){
			OUT.data_matrix[r][c]=Matrix_tag.data_matrix[r][c]-Matrix_out.data_matrix[r][c];
		}
	}
	return OUT;
}
Matrix label_processing(Matrix Data){ // label �B�z
	Matrix Matrix_D=matrix_equal(Data); // ���קK�ʨ��Data���ƭȡA�]���إߤ@�ӷs�Ŷ��h�sData���ƾ� 
	Matrix Label=matrix_get_col_label_data(Matrix_D,Data.data_col-1); // ���X���O����
	Matrix Label_1=one_hot_encoding(Matrix_D,Label); // one hot encoding�覡���s����
	return Label_1;
}
Matrix data_processing(Matrix Data){ // data �B�z
	Data=matrix_delete_last_col_data(Data); // �������O����
	Data=matrix_add_col_one(Data); //  ��J1�A�@��bais�ۭ��ɪ��v��
	return Data;
}
Net_layer create_net_layer(int data_num,int col,int net_num){ // �سy���g�h�A��J�`��ƾ��ӼơA����ӼơA���g���Ӽ�  
	Net_layer NetLayer;
	NetLayer.w=create_rand_matrix(net_num,col);  // �إ����üh����l�v���x�}  
	NetLayer.delta_w=create_zero_matrix(net_num,col);  // �إ߭ץ��v���x�} 
	NetLayer.net=create_zero_matrix(data_num,net_num);  // �[�`(�v��*�ƭ�)+bais
	NetLayer.net_sigmoid=create_zero_matrix(data_num,net_num);  // �Lsigmoid 
	NetLayer.error=create_zero_matrix(data_num,net_num); // layer��error 
	return NetLayer;
}
Matrix matrix_hidden_layer_error(Matrix weight,Matrix error){ // �˶ǻ��ɭp�����üh���~��function
	Matrix Matrix_sum=create_new_matrix(error.data_row,error.data_col);
	for(int c=0;c<Matrix_sum.data_col;c++){  //Herr_1=Oerr_1*O_1w+Oerr_1*O_2w+Oerr_1*O_3w
		for(int k=0;k<weight.data_col-1;k++){ 
			Matrix_sum.data_matrix[0][c]+=error.data_matrix[0][k]*weight.data_matrix[k][c];
		}
	}
	return Matrix_sum;
}
Matrix one_hot_encoding(Matrix data,Matrix label){  // ���쥻��data,��X��label 
	Matrix label_1=matrix_row_sort_small_to_large(label,0);// �N����{1,0}���ǧ令{0,1) 	
	Matrix goal_matrix=create_new_matrix(data.data_row,label_1.data_col);
	for(int i=0;i<data.data_row;i++){
		for(int j=0;j<goal_matrix.data_col;j++){ // ���]���ҬO012�Adata�����ҬO1�A�ഫ���G�O010 
			if(data.data_matrix[i][data.data_col-1]==label_1.data_matrix[0][j])
				goal_matrix.data_matrix[i][j]=1.0; // ���ҹ�����ɴN��1 
			else
				goal_matrix.data_matrix[i][j]=0.0; // �S������N��0 
		}
	}
	return goal_matrix;
}
NeuralNetwork net_forward(NeuralNetwork NN,Matrix Data){ // �e�V�Ǽ� 
	NN.H_layer.w=matrix_tran_last_col_negative(NN.H_layer.w); // �Nbais�ন�t�� 
	NN.H_layer.net=matrix_mult(Data,NN.H_layer.w); // �x�}���k�A���N�v����m 
	NN.H_layer.net_sigmoid=matrix_sigmoid(NN.H_layer.net); // hidden_net
	Matrix D=matrix_add_col_one(NN.H_layer.net_sigmoid); // �[�Fbais 
	NN.O_layer.w=matrix_tran_last_col_negative(NN.O_layer.w); // bais�ন�t��
	NN.O_layer.net=matrix_mult(D,NN.O_layer.w); // �x�}���k�A���N�v����m�A��XOutput_net
	NN.O_layer.net_sigmoid=matrix_sigmoid(NN.O_layer.net); // �w���� 
	return NN;	
} 
NeuralNetwork net_back(NeuralNetwork NN,Matrix Label){   // �˶ǻ� 
	NN.O_layer.error=matrix_loss_function_der(Label,NN.O_layer.net_sigmoid); // �q��X�h���^�� (T-Y) ����P��X�����~�t
	Matrix Sigmoid_der=matrix_sigmoid_der(NN.O_layer.net_sigmoid);  // Sigmoid �ɨ�� Y(1-Y)  ��X�h�w����sigmoid�ɨ��
	NN.O_layer.error=matrix_hadamard(NN.O_layer.error,Sigmoid_der); //�ϥΫ��F���n�x�}���k (T-Y)Y(1-Y) ��X�h�~�t
	Matrix Matrix_H_err=matrix_hidden_layer_error(NN.O_layer.w,NN.O_layer.error); // �v�������~�ۥ[
	NN.H_layer.error=matrix_sigmoid_der(NN.H_layer.net_sigmoid); //  ���üh�w����sigmoid�ɨ��
	NN.H_layer.error=matrix_hadamard(NN.H_layer.error,Matrix_H_err); //  ���üh�~�t
    return NN;
}
NeuralNetwork net_update_weight(NeuralNetwork NN,double learning_rate,Matrix Data){ // ��s�v�� 
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
NeuralNetwork net_update_bais(NeuralNetwork NN,double learning_rate){ // ��sbais 
	NN.H_layer.w=matrix_tran_last_col_negative(NN.H_layer.w); // �N�쥻��+bais �b��s����^ -bais 
	NN.O_layer.w=matrix_tran_last_col_negative(NN.O_layer.w); // �N�쥻��+bais �b��s����^ -bais  
	for(int r=0;r<NN.O_layer.delta_w.data_row;r++){  
		NN.O_layer.delta_w.data_matrix[r][NN.O_layer.delta_w.data_col-1]=-learning_rate*NN.O_layer.error.data_matrix[0][r];
	}
	for(int r=0;r<NN.H_layer.delta_w.data_row;r++){
		NN.H_layer.delta_w.data_matrix[r][NN.H_layer.delta_w.data_col-1]=-learning_rate*NN.H_layer.error.data_matrix[0][r];
	}
	NN.O_layer.w=matrix_plus(NN.O_layer.delta_w,NN.O_layer.w); // �~�t�v���[�W�v�� 
	NN.H_layer.w=matrix_plus(NN.H_layer.delta_w,NN.H_layer.w);
	NN.H_layer.delta_w=re_zero(NN.H_layer.delta_w); // �N�v���H�~�����k�s 
	NN.O_layer.delta_w=re_zero(NN.O_layer.delta_w);
	NN.H_layer.net=re_zero(NN.H_layer.net);
	NN.O_layer.net=re_zero(NN.O_layer.net);
	NN.H_layer.net_sigmoid=re_zero(NN.H_layer.net_sigmoid);
	NN.O_layer.net_sigmoid=re_zero(NN.O_layer.net_sigmoid);
	NN.H_layer.error=re_zero(NN.H_layer.error);
	NN.O_layer.error=re_zero(NN.O_layer.error);
	return NN;	
}
NeuralNetwork BGD_calculate_delta_weight(NeuralNetwork NN,double learning_rate,Matrix Data){ //�p��C���ƭȪ��v���Pbais�A�å����[�_�� 
	NN.H_layer.w=matrix_tran_last_col_negative(NN.H_layer.w); // �N�쥻��+bais(T-O�����Y) �b��s����^ -bais 
	NN.O_layer.w=matrix_tran_last_col_negative(NN.O_layer.w); // �N�쥻��+bais �b��s����^ -bais 
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
NeuralNetwork BGD_update_weight_and_bais(NeuralNetwork NN,int data_row){ // ��s�v���Pbais�A�n�N�[�`�����~�����Ⱥ�X�ӡA
//	double m=0.0;
//	m=1.0/data_row;
//	printf("m=%f \n",m);
//	NN.O_layer.delta_w=matrix_mult_num(NN.O_layer.delta_w,m); // ���Hdata�� 
//	printData(NN.O_layer.delta_w);
//	printData(NN.H_layer.delta_w);
	NN.O_layer.w=matrix_plus(NN.O_layer.delta_w,NN.O_layer.w); // �~�t�v���[�W�v�� 
	NN.H_layer.w=matrix_plus(NN.H_layer.delta_w,NN.H_layer.w);
	NN.H_layer.delta_w=re_zero(NN.H_layer.delta_w); // �N�v���H�~�����k�s 
	NN.O_layer.delta_w=re_zero(NN.O_layer.delta_w);	
	return NN;
} 
void save_weight(NeuralNetwork NN){ // �x�s���üh�B��X�h�v�� 
	ofstream writeFile;// �g���A�|�л\���쥻����T 
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
void save_nn_structure(NeuralNetwork NN,int hidden_net_num,int output_net_num,int data_col,double learning_rate,int iteration){ // �x�s���g�����[�c 
	ofstream writeFile;// �g���A�|�л\���쥻����T 
	writeFile.open("nn_structure.csv",ios::out);
	writeFile<<"input_net_num="<<data_col-1<<endl;
	writeFile<<"hidden_net_num="<<hidden_net_num<<endl;
	writeFile<<"output_net_num="<<output_net_num<<endl;
	writeFile<<"learning_rate="<<learning_rate<<endl;
	writeFile<<"hidden_layer_weight�G"<<endl;
	for(int r=0;r<NN.H_layer.w.data_row;r++){
		for(int c=0;c<NN.H_layer.w.data_col;c++){
			writeFile<<NN.H_layer.w.data_matrix[r][c]<<",";
		}
		writeFile<<endl;
	}
	writeFile<<"output_layer_weight�G"<<endl;
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
	NN.H_layer=create_net_layer(1,data_col,hidden_net_num); // �إ����üh 
	NN.O_layer=create_net_layer(1,hidden_net_num+1,output_net_num); // ��X�h�M���üh�x�}�B��ɡAcol�n�[�Wbais 
	Matrix Label=label_processing(Data); // label �B�z 
	printData(Label);
	Data=data_processing(Data); // data �B�z 
	while(iteration>0){ // �`������ 
		while(data_order<Data.data_row){ // �`���@������� 
			Matrix DATA=matrix_get_one_row_data(Data,data_order); // ���o�@����� 
			NN=net_forward(NN,DATA); // �e�V�Ǽ� 
			Matrix Label_1=matrix_get_one_row_data(Label,data_order); // ���o�P�C��label
			// �ˬd�w�����ҹ藍�� 
			forward_label=matrix_find_max_col(NN.O_layer.net_sigmoid);
			real_label=matrix_find_max_col(Label_1);
			if(forward_label!=real_label)
				cout_err_num++;	
			if(iteration==1) // �u�L�X�̫�@�����w�����G 
				printData(NN.O_layer.net_sigmoid);
			err_num=0.0;
			Matrix ERROR=matrix_loss_function(Label_1,NN.O_layer.net_sigmoid); // �p��error
			err_num=matrix_total(ERROR);
			if(err_num<stop_err) // �~�t�Ȱ������ 
				OK_data++;
//			if(iteration==1) //�p�G���������N���|�X�{�F 
//				printData(ERROR);
			NN=net_back(NN,Label_1); // �˶ǻ� 
			NN=net_update_weight(NN,learning_rate,DATA); // �p��ç�s�v�� 
			NN=net_update_bais(NN,learning_rate);  // �p��ç�sbais 
			data_order++;		
		}
//		printf("------iteration=%d------\n",iteration);
		if(OK_data==Data.data_row) // �p�G�������G�p����w�~�t����Ƥw�g�O�����F�A�N�� 
			break;
		else{
			err_rate=(double)cout_err_num/Data.data_row;
			if(iteration==1)
				printf("err_rate=%f\n",err_rate);
			cout_err_num=0;
			OK_data=0;   // �����V�mOK���ƶq�k�s 
			data_order=0; // �q�Y�V�m 
			iteration--;  // ���N���ƴ� 1  
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
			printf("�п�J���ո�ơG");
			while( !( cin >> number ) ){
				cin.clear(); //�M�� ios_base::failbit
				getline( cin, str ); //�M���@��
				cout << "�п�J�Ʀr" << endl;
		  	}
			testData.data_matrix[0][c]=number;
			cout<<testData.data_matrix[0][c]<<endl;
			c++;
			feature--;
		}
		printData(NN.H_layer.w);
		NN=net_forward(NN,testData); // �e�V�Ǽ� 
		printf("�w�����G���G\n");
		printData(NN.O_layer.net_sigmoid);	
		printf("�O�_�~��w��? y/n \n");
		cin>>ans;	
		NN.H_layer.w=matrix_tran_last_col_negative(NN.H_layer.w); // bais�ন�t��
		NN.O_layer.w=matrix_tran_last_col_negative(NN.O_layer.w); // bais�ন�t��
	}
}
void BGD(Matrix Data,int hidden_net_num,int output_net_num,int data_col,double learning_rate,int iteration,double stop_err){ // �H����פU�� 
	int data_order=0,OK_data=0;
	double err_num;
	NeuralNetwork NN;
	NN.H_layer=create_net_layer(1,data_col,hidden_net_num); // �إ����üh 
	NN.O_layer=create_net_layer(1,hidden_net_num+1,output_net_num); // ��X�h�M���üh�x�}�B��ɡAcol�n�[�Wbais 
	Matrix Label=label_processing(Data); // label �B�z 
	Data=data_processing(Data); // data �B�z 
	while(iteration>0){ // �`������ 
		while(data_order<Data.data_row){ // �`���@������� 
			Matrix DATA=matrix_get_one_row_data(Data,data_order); // ���o�@����� 
			NN=net_forward(NN,DATA); // �e�V�Ǽ� 
//			printData(DATA);
//			printData(NN.O_layer.net_sigmoid)
			Matrix Label_1=matrix_get_one_row_data(Label,data_order); // ���o�P�C��label 
//			printData(Label_1);
			if(iteration==1) // �u�L�X�̫�@�����w�����G 
				printData(NN.O_layer.net_sigmoid);
			err_num=0.0;	
			Matrix ERROR=matrix_loss_function(Label_1,NN.O_layer.net_sigmoid); // �p��error
			err_num=matrix_total(ERROR);
			if(err_num<stop_err*2)
				OK_data++;
//			if(iteration==1)
//				printData(ERROR);
			NN=net_back(NN,Label_1); // �˶ǻ� 
//			printALLData(NN);
//			printf("===back----------back===\n");
			NN=BGD_calculate_delta_weight(NN,learning_rate,DATA); //�p��C���ƭȪ��v���Pbais�å[�_�� 
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
			printf("�п�J���ո�ơG");
			while( !( cin >> number ) ){
				cin.clear(); //�M�� ios_base::failbit
				getline( cin, str ); //�M���@��
				cout << "�п�J�Ʀr" << endl;
		  	}
//			if(number)
			testData.data_matrix[0][c]=number;
			cout<<testData.data_matrix[0][c]<<endl;
			printData(testData);
			c++;
			feature--;
			
		}
		printData(testData);
		NN=net_forward(NN,testData); // �e�V�Ǽ� 
		printf("�w�����G���G\n");
		printData(NN.O_layer.net_sigmoid);	
		printf("�O�_�~��w��? y/n \n");
		cin>>ans;	
		NN.H_layer.w=matrix_tran_last_col_negative(NN.H_layer.w); // bais�ন�t��
		NN.O_layer.w=matrix_tran_last_col_negative(NN.O_layer.w); // bais�ন�t��
	}
} 
