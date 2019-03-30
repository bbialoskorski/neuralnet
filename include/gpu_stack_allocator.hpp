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

#ifndef NEURALNET_INCLUDE_GPU_STACK_ALLOCATOR_HPP_
#define NEURALNET_INCLUDE_GPU_STACK_ALLOCATOR_HPP_

#include "gpu_allocation_manager.hpp"

#include "unordered_map"

namespace neuralnet {

/**
 * @brief Efficient gpu memory stack allocator which preallocates big block of
 *        memory at the beggining and then just manages pointers to that block.
 *
 * When using this allocator user has to free memory in reverse order to
 * allocation order to make sure he's popping from the top of a stack.
 */
class GpuStackAllocator : public GpuAllocationManager {
 public:
  /**
   * @brief Creates stack allocator and preallocates specified amount of gpu
   *        memory.
   *
   * @param size Amount of memory to preallocate in bytes.
   */
  GpuStackAllocator(size_t size);
  ~GpuStackAllocator();
  
  void* AllocateDevice(size_t size) override;
  void FreeDevice(void* d_ptr) override;
 private:
  std::unordered_map<void*, std::pair<size_t, long long>> allocated_blocks_;
  size_t memory_size_;
  void* d_memory_block_;
  void* d_top_;
  void* d_end_;
};

} // namespace neuralnet

#endif // !NEURALNET_INCLUDE_GPU_STACK_ALLOCATOR_HPP_

