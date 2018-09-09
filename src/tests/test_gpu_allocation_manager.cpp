#include "gpu_allocation_manager.hpp"

#include "gtest/gtest.h"

#include "cuda_runtime.h"

namespace neuralnet_src_tests {

class AllocationManagerTest : public testing::Test {
 protected:
  neuralnet::GpuAllocationManager mem_manager_;
};

TEST_F(AllocationManagerTest,
       GetLastAllocationIdValueWithoutPreviousAllocations) {
  long long allocation_id = mem_manager_.GetLastAllocationId();
  EXPECT_EQ(-1, allocation_id);
}

TEST_F(AllocationManagerTest, AllocationIdInitialValue0) {
  long long allocation_id = mem_manager_.GetLastAllocationId() + 1;
  EXPECT_EQ(0, allocation_id);
}

TEST_F(AllocationManagerTest, ManagesIntegerAllocation) {
  int* d_int = (int*)mem_manager_.AllocateDevice(sizeof(int));
  mem_manager_.FreeDevice(d_int);
}

TEST_F(AllocationManagerTest, ManagesFloatAllocation) {
  float* d_float = (float*)mem_manager_.AllocateDevice(sizeof(float));
  mem_manager_.FreeDevice(d_float);
}

TEST_F(AllocationManagerTest, ManagesDoubleAllocation) {
  double* d_double = (double*)mem_manager_.AllocateDevice(sizeof(double));
  mem_manager_.FreeDevice(d_double);
}

TEST_F(AllocationManagerTest, EmbedsAllocationIdWithinAllocatedBlock) {
  int* d_int = (int*)mem_manager_.AllocateDevice(sizeof(int));
  long long alloc_id;
  cudaMemcpy(&alloc_id, (char*)d_int - sizeof(long long),
            sizeof(long long), cudaMemcpyDeviceToHost);
  EXPECT_EQ(mem_manager_.GetLastAllocationId(), alloc_id);
}

TEST_F(AllocationManagerTest, IncrementsAllocationId) {
  long long alloc_id;
  int* d_int = (int*)mem_manager_.AllocateDevice(sizeof(int));
  alloc_id = mem_manager_.GetLastAllocationId();
  float* d_float = (float*)mem_manager_.AllocateDevice(sizeof(float));
  ASSERT_EQ(alloc_id + 1, mem_manager_.GetLastAllocationId());
  alloc_id = mem_manager_.GetLastAllocationId();
  double* d_double = (double*)mem_manager_.AllocateDevice(sizeof(double));
  ASSERT_EQ(alloc_id + 1, mem_manager_.GetLastAllocationId());
  mem_manager_.FreeDevice(d_int);
  mem_manager_.FreeDevice(d_float);
  mem_manager_.FreeDevice(d_double);
}

} // namespace neuralnet_src_tests