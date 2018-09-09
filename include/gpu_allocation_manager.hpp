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

#ifndef NEURALNET_INCLUDE_GPU_ALLOCATION_MANAGER_HPP_
#define NEURALNET_INCLUDE_GPU_ALLOCATION_MANAGER_HPP_

#include <atomic>
#include <unordered_set>

namespace neuralnet {

/**
 * @brief Wrapper around cudaMalloc() and cudaFree() tracking unfreed device
 *        memory allocations.
 */
class GpuAllocationManager {
 public:
  virtual ~GpuAllocationManager() {}
  /**
   * @brief Returns id of most recent allocation or -1 if no allocations were
   *        made.
   */
  long long GetLastAllocationId();
  /**
   * @brief Allocates size_t bytes on gpu device.
   *
   * @param size Number of bytes to allocate.
   * @returns    Pointer to allocated device memory block.
   * @throws std::runtime_error If allocation failed.
   */
  void* AllocateDevice(size_t size);
  /**
   * @brief Frees device memory block.
   *
   * @param d_ptr Pointer to the beggining of device memory block you want to
   *              free.
   * @throws std::runtime_error If freeing device memory failed.
   */
  void FreeDevice(void* d_ptr);
  /** @brief Prints out unfreed allocations ids. */
  void PrintAllocationState();

 protected:
  static long long allocation_id_;
  /**< Id of the next allocation. */
  static std::unordered_set<long long> allocated_set_;
  /**< Set containing ids of unfreed device memory blocks. */
};

}  // namespace neuralnet

#endif  // !NEURALNET_INCLUDE_GPU_ALLOCATION_MANAGER_HPP_
