/* Copyright (c) 2018 Bartosz Białoskórski

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#include "gpu_allocation_manager.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

#include "cuda_runtime.h"

namespace neuralnet {

long long GpuAllocationManager::allocation_id_ = 0;
std::unordered_set<long long> GpuAllocationManager::allocated_set_;

long long GpuAllocationManager::GetLastAllocationId() {
  return allocation_id_ - 1;
}

void* GpuAllocationManager::AllocateDevice(size_t size) {
  long long id = allocation_id_++;
  allocated_set_.insert(id);
  void* d_ptr;
  cudaError_t cuda_status;
  cuda_status = cudaMalloc(&d_ptr, size + sizeof(long long));

  if (cuda_status != cudaSuccess) {
    std::string err_msg =
        "Error (" + std::string(cudaGetErrorString(cuda_status)) +
        ") occured while trying to allocate memory on device (allocation id " +
        std::to_string(id) + ")";
    throw std::runtime_error(err_msg);
  }

  cuda_status =
      cudaMemcpy(d_ptr, &id, sizeof(long long), cudaMemcpyHostToDevice);

  if (cuda_status != cudaSuccess) {
    std::string err_msg =
        "Error (" + std::string(cudaGetErrorString(cuda_status)) +
        ") occured while trying to copy allocation id to allocated block \
(allocation id " +
        std::to_string(id) + ")";

    throw std::runtime_error(err_msg);
  }

  return (char*)d_ptr + sizeof(long long);
}

void GpuAllocationManager::FreeDevice(void* d_ptr) {
  long long id;
  cudaError_t cuda_status;
  cuda_status = cudaMemcpy(&id, (char*)d_ptr - sizeof(long long),
                           sizeof(long long), cudaMemcpyDeviceToHost);

  if (cuda_status != cudaSuccess) {
    std::string err_msg =
        "Error (" + std::string(cudaGetErrorString(cuda_status)) +
        ") occured while trying to acquire allocation id from device \
pointer.";
    throw std::runtime_error(err_msg);
  }

  cuda_status = cudaFree((char*)d_ptr - sizeof(long long));

  if (cuda_status != cudaSuccess) {
    std::string err_msg =
        "Error (" + std::string(cudaGetErrorString(cuda_status)) +
        ") occured while trying to free memory from device (allocation id " +
        std::to_string(id) + ")";

    throw std::runtime_error(err_msg);
  }

  allocated_set_.erase(id);
}

void GpuAllocationManager::PrintAllocationState() {
  std::cout << "Allocations not freed:";
  for (long long id : allocated_set_) std::cout << " " << id;
  std::cout << std::endl;
}

}  // namespace neuralnet
