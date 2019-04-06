/* Copyright (c) 2019 Bartosz Białoskórski

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

#include "gpu_stack_allocator.hpp"

#include <iostream>

#include "cuda_runtime.h"

namespace neuralnet {

GpuStackAllocator::GpuStackAllocator(size_t preallocated_block_size) {
  memory_size_ = preallocated_block_size;

  cudaError_t cuda_status;
  cuda_status = cudaMalloc((void**)&d_memory_block_, preallocated_block_size);
  if (cuda_status != cudaSuccess) {
    std::string err_msg = "Cuda failed to allocate memory block of requested\
 size";
    throw std::runtime_error(err_msg);
  }

  d_top_ = d_memory_block_;
  d_end_ = (char*)d_memory_block_ + preallocated_block_size;
}

GpuStackAllocator::~GpuStackAllocator() {
  cudaFree(d_memory_block_);
}

void* GpuStackAllocator::AllocateDevice(size_t size) {
  if ((char*)d_top_ + size <= d_end_) {
    void* d_ptr = d_top_;
    long long id = allocation_id_++;
    allocated_blocks_.insert(std::make_pair(d_ptr, std::make_pair(size, id)));
    allocated_ids_.insert(id);
    // Moving pointer to top of the stack up.
    d_top_ = (char*)d_top_ + size;
    return d_ptr;
  }
  else {
    // Not enough space on the stack for this allocation.
    std::string err_msg = "GpuStackAllocator can't allocate memory because\
 stack is full.";
    throw std::runtime_error(err_msg);
  }
}

void GpuStackAllocator::FreeDevice(void* d_ptr) {
  std::pair<size_t, long long> block_info;
  try {
    block_info = allocated_blocks_.at(d_ptr);
    if ((char*)d_top_ - block_info.first != d_ptr) {
      std::string err_msg = "Error - attempt to deallocate from middle of the\
 stack.";
      throw std::runtime_error(err_msg);
    }
    // Moving pointer to top of the stack down.
    d_top_ = (char*)d_top_ - block_info.first;
    allocated_blocks_.erase(d_ptr);
    allocated_ids_.erase(block_info.second);
  }
  catch (std::out_of_range& exception) {
    std::string err_msg = "Trying to free memory not allocated by gpu stack\
 allocator.";
    throw std::runtime_error(err_msg);
  }
}

} // namespace neuralnet