#include<iostream>
#include<fstream>
#include<string>
#include<stdio.h>
#include<math.h>

using namespace std;
void readfile(const char* filename, float a[], int& n) {
	ifstream input(filename);
	float k;
	n = 0;
	while (input >> k) {
		a[n++] = k;
	}
	input.close();
}
void writefile1d(const char* filename, float a[], int n) {
	ofstream output(filename);
	for (int i = 0; i < n; i++) {
		output << a[i] << " ";
	}
	output << endl;
}
void xuat(float a[], int n) {
	for (int i = 0; i < n; i++) {
		printf("a[%d] = %0.15f ", i, a[i]);
	}
	printf("\n");
}
void xuatmang2d(float** a, int row, int col) {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			printf("out2d[%d][%d] = %f \n", i, j, a[i][j]);
		}
	}
}
void transpose2d(float a[], int n, float** out, int row, int col) {
	int k = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			out[i][j] = a[k++];
		}
	}
	xuatmang2d(out, row, col);
}
void sigmoid(float x[], int n, float out[]) {
	for (int i = 0; i < n; i++) {
		out[i] = 1 / (1 + exp(-x[i]));
	}
}
void softmax(float* input, int n_input, float* output) {
	float max_val = input[0];
	for (int i = 0; i < n_input; i++) {
		if (input[i] > max_val) {
			max_val = input[i];
		}
	}
	float exp_sum = 0.0;
	for (int i = 0; i < n_input; i++) {
		float exp_val = expf(input[i] - max_val);
		output[i] = exp_val;
		exp_sum += exp_val;
	}
	for (int i = 0; i < n_input; i++) {
		output[i] /= exp_sum;
	}
}
void conv2d(float* input, float* w, int row, int col, int frow, int fcol, int channel_inputs,
	int kernel_size, float* conv_results, float* bias) {
	//Allocate memory for kernel_out dynamically
	float**** kernel_out = (float****)malloc(frow * sizeof(float***));
	for (int i = 0; i < frow; i++) {
		kernel_out[i] = (float***)malloc(fcol * sizeof(float**));
		for (int j = 0; j < fcol; j++) {
			kernel_out[i][j] = (float**)malloc(channel_inputs * sizeof(float*));
			for (int k = 0; k < channel_inputs; k++) {
				kernel_out[i][j][k] = (float*)malloc(kernel_size * sizeof(float));
			}
		}
	}
	int s = 0;
	for (int i = 0; i < frow; i++) {
		for (int j = 0; j < fcol; j++) {
			for (int k = 0; k < channel_inputs; k++) {
				for (int l = 0; l < kernel_size; l++)
				{
					kernel_out[i][j][k][l] = w[s++];
				}
			}
		}
	}
	for (int k = 0; k < kernel_size; k++) {
		for (int r = 0; r < row - frow + 1; r++) {
			for (int c = 0; c < col - fcol + 1; c++) {
				float sum = 0;
				for (int ch = 0; ch < channel_inputs; ch++) {
					for (int m = r; m < r + frow; m++) {
						for (int n = c; n < c + fcol; n++) {
							sum += input[(m * col + n) * channel_inputs + ch] * kernel_out[m - r][n - c][ch][k];
						}
					}
				}
				conv_results[r * (col - fcol + 1) * kernel_size + c * kernel_size + k] = sum + bias[k];
			}
		}
	}
	//Free memory for kernel_out
	for (int i = 0; i < frow; i++) {
		for (int j = 0; j < fcol; j++) {
			for (int k = 0; k < channel_inputs; k++) {
				free(kernel_out[i][j][k]);
			}
			free(kernel_out[i][j]);
		}
		free(kernel_out[i]);
	}
	free(kernel_out);
	//for (int i = 0; i < (row - frow + 1) * (col - fcol + 1) * kernel_size; i++) {
	//	printf("out_conv2d[%d] = %0.10f ", i, conv_results[i]);
	//}
}

void maxpool2d(float* input, int row, int col, int pool_size, int channel_size, int stride, float* output_max) {
	int output_row = ((row - pool_size) / stride) + 1;
	int output_col = ((col - pool_size) / stride) + 1;

	for (int k = 0; k < channel_size; k++) {
		for (int r = 0; r < output_row; r++) {
			for (int c = 0; c < output_col; c++) {
				float max_val = -INFINITY;
				for (int m = r * stride; m < r * stride + pool_size; m++) {
					for (int n = c * stride; n < c * stride + pool_size; n++) {
						float val = input[(m * col + n) * channel_size + k];
						if (val > max_val) {
							max_val = val;
						}
					}
				}
				output_max[(r * output_col + c) * channel_size + k] = max_val;
			}
		}
	}
	//for (int i = 0; i < output_row * output_col * channel_size; i++) {
	//	printf("out[%d]= %0.10f ", i, output_max[i]);
	//}
}
void dense(float* input, int n_input, int n_w, float* weights, float* bias, float* output) {

	float* dot = (float*)malloc(n_w * sizeof(float));

	for (int i = 0; i < n_w; i++) {
		dot[i] = 0;
		for (int j = 0; j < n_input; j++) {
			dot[i] += input[j] * weights[j * n_w + i];
		}
		output[i] = dot[i] + bias[i];
	}

	//for (int i = 0; i < n_w; i++) {
	//	printf("output[%i] = %f \n", i, output[i]);
	//}

	free(dot);
}
void main() {
	//input X_test[0] = 7
	float input[28 * 28];
	int n_input;
	readfile("so_2.txt", input, n_input);
	//xuat(input, n_input);
	//===========================conv2d_0============================
	float w_0[288];
	float bias_0[32];
	int nw_0;
	int nb_0;
	int l_0 = 26 * 26 * 32;
	float out_0[26 * 26 * 32];
	float out_sig_0[26 * 26 * 32];
	readfile("weights_0.txt", w_0, nw_0);
	readfile("bias_0.txt", bias_0, nb_0);
	conv2d(input, w_0, 28, 28, 3, 3, 1, 32, out_0, bias_0);
	sigmoid(out_0, l_0, out_sig_0);
	//===========================conv2d_1=============================
	//float w_1[9216];
	float* w_1 = (float*)malloc(9216 * sizeof(float));
	float bias_1[32];
	int nw_1;
	int nb_1;
	int l_1 = 24 * 24 * 32;
	float out_1[24 * 24 * 32];
	float out_sig_1[24 * 24 * 32];
	readfile("weights_1.txt", w_1, nw_1);
	readfile("bias_1.txt", bias_1, nb_1);
	conv2d(out_sig_0, w_1, 26, 26, 3, 3, 32, 32, out_1, bias_1);
	sigmoid(out_1, l_1, out_sig_1);
	//============================maxpool2d============================
	float out_maxpool_0[12 * 12 * 32];
	maxpool2d(out_sig_1, 24, 24, 2, 32, 2, out_maxpool_0);
	//============================Dense128=============================
	float* w_128 = (float*)malloc(589824 * sizeof(float));
	float bias_128[128];
	int nw_128;
	int nb_128;
	int l_dense_128 = 12 * 12 * 32;
	int l_w_128 = 128;
	float out_dense_128[128];
	float out_sig_128[128];
	readfile("dense_128.txt", w_128, nw_128);
	readfile("dense_bias_128.txt", bias_128, nb_128);
	dense(out_maxpool_0, l_dense_128, l_w_128, w_128, bias_128, out_dense_128);
	sigmoid(out_dense_128, 128, out_sig_128);
	//=============================Dense10=============================
	float w_10[1280];
	float bias_10[10];
	int nw_10;
	int nb_10;
	int l_dense_10 = 128;
	int l_w_10 = 10;
	float out_dense_10[10];
	float out_softmax_10[10];
	readfile("dense_10.txt", w_10, nw_10);
	readfile("dense_bias_10.txt", bias_10, nb_10);
	dense(out_sig_128, l_dense_10, l_w_10, w_10, bias_10, out_dense_10);
	softmax(out_dense_10, 10, out_softmax_10);
	xuat(out_softmax_10, 10);
	free(w_1);
	free(w_128);
}