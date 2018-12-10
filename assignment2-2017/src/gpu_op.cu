#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define NBLOCKS 10
#define NTHREADS 10

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}


__global__ void array_set_kernel(float value, float *input, int nrow, int row_size, int unit_size){
    int index = threadIdx.x + blockIdx.x * blockDim.x; 
    int start = index * unit_size;
    int end = min(nrow, start + unit_size);
    if(start >= nrow)
        return;
    else{
        for(int ii = start; ii < end; ii++)
            for(int jj = 0; jj < row_size; jj++)
                *((input + ii * row_size) + jj) = value;
    }
    return;
}


int DLGpuArraySet(DLArrayHandle arr, float value) { /* TODO: Your code here */
    int nrow = arr->shape[0];
    int row_size = 1;
    for(int d=1; d<arr->ndim; d++) row_size *= arr->shape[d];

    int unit_size = nrow / (NBLOCKS * NTHREADS) + 1;
    array_set_kernel<<<NBLOCKS, NTHREADS>>>(value, (float *)arr->data, nrow, row_size, unit_size);

  return 0;
}

__global__ void array_broadcastto_kernel(float *input, float *output, int nrow, int row_size, int unit_size){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int start = index * unit_size;
    int end = min(nrow, start + unit_size);
    if(start >= nrow) return;
    else{
        for(int ii = start; ii < end; ii++)
            for(int jj = 0; jj < row_size; jj++)
                *(output + ii * row_size + jj) = *(input + jj);
    }
    return;
}
int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
    int nrow = output->shape[0];
    int row_size = 1;
    for(int d = 0; d < input->ndim; d++) row_size *= input->shape[d];
    
    int unit_size = nrow / (NBLOCKS * NTHREADS) + 1;
    array_broadcastto_kernel<<<NBLOCKS, NTHREADS>>>((float *)input->data, (float *)output->data, nrow, row_size, unit_size);
  return 0;
}

__global__ void array_reduceSumAxisZero(float *input, float *output, int nrow, int row_size, int ncol, int col_size, int unit_size){
    /* parallelize over col */
   int index = threadIdx.x + blockIdx.x * blockDim.x;
   int start = index * unit_size;
   int end = min(start + unit_size, ncol);
   if(start >= ncol) return;
   else{ 
       for(int jj = start * col_size; jj < end * col_size; jj++)
           for(int rowId = 0; rowId < nrow; rowId ++)
               *(output + jj) += *(input + rowId * row_size + jj);
   }
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
    /* parallelize over second dimension */
    int nrow = input->shape[0];
    int ncol = input->shape[1];
    int col_size = 1;
    for(int d = 2; d < input->ndim; d++) col_size *= input->shape[d];
    int row_size = col_size * input->shape[1];

    int unit_size = ncol / (NBLOCKS *NTHREADS) + 1;
    array_reduceSumAxisZero<<<NBLOCKS, NTHREADS>>>((float *)input->data, (float *)output->data, nrow, row_size, ncol, col_size, unit_size);
  return 0;
}

__global__ void array_matEleAdd(float *A, float *B, float *output, int nrow, int row_size, int unit_size){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int start = index * unit_size;
    int end = min(nrow, start + unit_size);
    if(start >= nrow) return;
    else{
        for(int rowId = start; rowId < end; rowId ++)
            for(int jj = 0; jj < row_size; jj ++){
                int bias = rowId * row_size + jj;
                *(output + bias) = *(A + bias) + *(B + bias);
            }
    }
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  /* TODO: Your code here */
    int nrow = matA->shape[0];
    int row_size = 1;
    for(int d = 1; d < matA->ndim; d++) row_size *= matA->shape[d];

    int unit_size = nrow / (NBLOCKS * NTHREADS) + 1;
    array_matEleAdd<<<NBLOCKS, NTHREADS>>>((float *)matA->data, (float *)matB->data, (float *)output->data, nrow, row_size, unit_size);
  return 0;
}

__global__ void array_matEleAddByConst(float *input, float *output, float val, int nrow, int row_size, int unit_size){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int start = index * unit_size;
    int end = min(nrow, start + unit_size);
    if(start >= nrow) return;
    else{
        for(int rowId = start; rowId < end; rowId ++)
            for(int jj = 0; jj < row_size; jj ++){
                int bias = rowId * row_size + jj;
                *(output + bias) = *(input + bias) + val;
            }
    }
}
int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  /* TODO: Your code here */
    int nrow = input->shape[0];
    int row_size = 1;
    for(int d = 1; d < input->ndim; d++) row_size *= input->shape[d];

    int unit_size = nrow / (NBLOCKS * NTHREADS) + 1;
    array_matEleAddByConst<<<NBLOCKS, NTHREADS>>>((float *)input->data, (float *) output->data, val, nrow, row_size, unit_size);
  return 0;
}

__global__ void array_matEleMult(float *A, float *B, float *output, int nrow, int row_size, int unit_size){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int start = index * unit_size;
    int end = min(nrow, start + unit_size);
    if(start >= nrow) return;
    else{
        for(int rowId = start; rowId < end; rowId ++)
            for(int jj = 0; jj < row_size; jj ++){
                int bias = rowId * row_size + jj;
                *(output + bias) = *(A + bias) * *(B + bias);
            }
    }
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  /* TODO: Your code here */
    int nrow = matA->shape[0];
    int row_size = 1;
    for(int d = 1; d < matA->ndim; d++) row_size *= matA->shape[d];

    int unit_size = nrow / (NBLOCKS * NTHREADS) + 1;
    array_matEleMult<<<NBLOCKS, NTHREADS>>>((float *)matA->data, (float *)matB->data, (float *)output->data, nrow, row_size, unit_size);
  return 0;
}

__global__ void array_matEleMultByConst(float *input, float *output, float val, int nrow, int row_size, int unit_size){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int start = index * unit_size;
    int end = min(nrow, start + unit_size);
    if(start >= nrow) return;
    else{
        for(int rowId = start; rowId < end; rowId ++)
            for(int jj = 0; jj < row_size; jj ++){
                int bias = rowId * row_size + jj;
                *(output + bias) = *(input + bias) * val;
            }
    }
}
int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  /* TODO: Your code here */
    int nrow = input->shape[0];
    int row_size = 1;
    for(int d = 1; d < input->ndim; d++) row_size *= input->shape[d];

    int unit_size = nrow / (NBLOCKS * NTHREADS) + 1;
    array_matEleMultByConst<<<NBLOCKS, NTHREADS>>>((float *)input->data, (float *)output->data, val, nrow, row_size, unit_size);
  return 0;
}

__global__ void array_relu(float *input, float *output, int nrow, int row_size, int unit_size){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int start = index * unit_size;
    int end = min(nrow, start + unit_size);
    if(start >= nrow) return;
    else{
        for(int rowId = start; rowId < end; rowId ++)
            for(int jj = 0; jj < row_size; jj ++){
                int bias = rowId * row_size + jj;
                *(output + bias) = max(0., *(input + bias));
            }
    }
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
    int nrow = input->shape[0];
    int row_size = 1;
    for(int d = 1; d < input->ndim; d ++) row_size *= input->shape[d];

    int unit_size = nrow / (NBLOCKS * NTHREADS) + 1;
    array_relu<<<NBLOCKS, NTHREADS>>>((float *)input->data, (float *)output->data, nrow, row_size, unit_size);
  return 0;
}

__global__ void array_reluGrad(float *input, float *grad, float *output, int nrow, int row_size, int unit_size){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int start = index * unit_size;
    int end = min(nrow, start + unit_size);
    if(start >= nrow) return;
    else{
        for(int rowId = start; rowId < end; rowId ++)
            for(int jj = 0; jj < row_size; jj ++){
                int bias = rowId * row_size + jj;
                int sign = 0;
                if(*(input + bias) > 0) sign = 1;
                else if(*(input + bias) < 0) sign = -1;
                else sign = 0;
                *(output + bias) = (sign + 1) * 0.5 * (*(grad + bias));
            }
    }
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  /* TODO: Your code here */
    int nrow = input->shape[0];
    int row_size = 1;
    for(int d = 1; d < input->ndim; d ++) row_size *= input->shape[d];

    int unit_size = nrow / (NBLOCKS * NTHREADS) + 1;
    array_reluGrad<<<NBLOCKS, NTHREADS>>>((float *)input->data, (float *)in_grad->data, (float *)output->data, nrow, row_size, unit_size);
  return 0;
}

__global__ void array_softmax(float *input, float *output, int nrow, int row_size, int unit_size){
    
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int start = index * unit_size;
    int end = min(nrow, start + unit_size);
    if(start >= nrow) return;
    else{
        for(int rowId = start; rowId < end; rowId ++){
            double _expsum = 0; // sum of this row
            double _max = FLT_MIN; // max in this row
            int bias;
            for(int jj = 0; jj < row_size; jj ++){
                bias = rowId * row_size + jj;
                _max = max(_max, *(input + bias));
            }
            for(int jj = 0; jj < row_size; jj ++){
                bias = rowId * row_size + jj;
                _expsum += expf(*(input + bias) - _max);
            }
            for(int jj = 0; jj < row_size; jj ++){
                bias = rowId * row_size + jj;
                *(output + bias) = expf(*(input + bias) - _max) / _expsum;
            }
        }
    }
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
    assert(input->ndim == 2);
    int nrow = input->shape[0];
    int row_size = 1;
    for(int d = 1; d < input->ndim; d ++) row_size *= input->shape[d];

    int unit_size = nrow / (NBLOCKS * NTHREADS) + 1;
    array_softmax<<<NBLOCKS, NTHREADS>>>((float *)input->data, (float *)output->data, nrow, row_size, unit_size);
  return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  /* TODO: Your code here */
  // Hint: use cublas
  // cublas assume matrix is column major
  return 0;
}
