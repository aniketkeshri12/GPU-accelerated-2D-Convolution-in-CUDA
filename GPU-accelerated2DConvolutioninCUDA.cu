/**
*   CS6023: GPU Programming 
*   Assignment 2
*   
*   Please don't change any existing code in this file.
*
*   Please add necessary memory APIs for your implementation. Use cudaFree() 
*   to free up memory as soon as you're done with an allocation. 
*   This will ensure that you don't run out of memory while running
*   large test cases. Use the minimum required memory for your 
*   implementation. DO NOT change the kernel configuration parameters.
*/

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;

 //kernel for convolution operation
__global__ void convolutionKernel(long int* d_mat, long int* d_filter, long int* d_ans, long int m, long int n, long int k) {

    extern __shared__ long int shared_filter[]; //dynamic shared memory for optimization 

    // Loading filter into shared memory
    if (threadIdx.x == 0)
    {
        for (long int i = 0; i < k * k; i++)
        {
            shared_filter[i] = d_filter[i]; //original filter is copied in shared memory
        }
    }

    __syncthreads();
    long int id = blockIdx.x * blockDim.x + threadIdx.x; //Id for particular thread

    long int row = id / n; // Each block corresponds to one row of the output matrix
    long int col = id % n; // Each thread corresponds to one column of the output matrix


    long int padding = k / 2; //padding calculation 
    long int sum = 0;


    for (long int a = -padding; a <= padding; a++) {
        for (long int b = -padding; b <= padding; b++) {

            long int nx = row + a;
            long int ny = col + b;

            if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                sum += d_mat[nx * n + ny] * shared_filter[(a + padding) * k + (b + padding)];
            }

        }
    }

    d_ans[row * n + col] = sum; //will store final answer 

}

int main(int argc, char** argv) {

    int m,n,k;
    cin>>m>>n>>k;


    long int* h_mat = new long int[m * n];
    long int* h_filter = new long int[k * k];

    long int* h_ans = new long int[m * n];


    for (long int i = 0; i < m * n; i++) {
        cin>>h_mat[i];
    }

    for (long int i = 0; i < k * k; i++) {
        cin>>h_filter[i];
    }

    /**
     * 
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     * 
    **/

    /****************************************************Start Here***********************************************************/
    
     //Allocated device memory
    long int* d_mat, * d_filter, * d_ans;
    cudaMalloc(&d_mat, m * n * sizeof(long int));
    cudaMalloc(&d_filter, k * k * sizeof(long int));
    cudaMalloc(&d_ans, m * n * sizeof(long int));

    //copy data from host to device 
    cudaMemcpy(d_mat, h_mat, m * n * sizeof(long int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, k * k * sizeof(long int), cudaMemcpyHostToDevice);

    //Define block and grid dimentions

    auto start = std::chrono::high_resolution_clock::now();//keep it just before the kernel launch
     
    // Launch kernel
    convolutionKernel << <m, n, k* k * sizeof(long int) >> > (d_mat, d_filter, d_ans, m, n, k);


    auto end = std::chrono::high_resolution_clock::now();//keep it just after the kernel launch
    
    //copy result back to host from device
    cudaMemcpy(h_ans, d_ans, m * n * sizeof(long int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    
    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

    cudaFree(d_mat);
    cudaFree(d_filter);
    cudaFree(d_ans);
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     * 
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     * 
    */


    
    std::ofstream file("cuda.out");
    if (file.is_open()) {
        for (long int i = 0; i < m; i++) {
            for (long int j = 0; j < n; j++) {
                file << h_ans[i * n + j] << " ";
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if(file2.is_open()) {
        file2 << elapsed1.count() << "\n";
        file2.close();
    } else {
        std::cout << "Unable to open file";
    }

    return 0;
}