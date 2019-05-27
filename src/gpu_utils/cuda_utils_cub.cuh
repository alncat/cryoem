#ifndef CUDA_UTILS_CUB_CUH_
#define CUDA_UTILS_CUB_CUH_

#include <cuda_runtime.h>
#include "src/gpu_utils/cuda_settings.h"
#include "src/gpu_utils/cuda_mem_utils.h"
#include <stdio.h>
#include <limits>
#include <signal.h>
#include <vector>
#include <math.h>
// Because thrust uses CUB, thrust defines CubLog and CUB tries to redefine it,
// resulting in warnings. This avoids those warnings.
#if(defined(CubLog) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__<= 520)) // Intetionally force a warning for new arch
	#undef CubLog
#endif
//#include "src/gpu_utils/cub/cub.cuh"
#include "src/gpu_utils/cub/block/block_reduce.cuh"
//#include "src/gpu_utils/cub/block/block_histogram.cuh"
//#include "src/gpu_utils/cub/block/block_discontinuity.cuh"
//#include "src/gpu_utils/cub/block/block_exchange.cuh"
#include "src/gpu_utils/cub/block/block_load.cuh"
//#include "src/gpu_utils/cub/block/block_radix_rank.cuh"
//#include "src/gpu_utils/cub/block/block_radix_sort.cuh"
//#include "src/gpu_utils/cub/block/block_scan.cuh"
#include "src/gpu_utils/cub/block/block_store.cuh"
//
#include "src/gpu_utils/cub/device/device_radix_sort.cuh"
#include "src/gpu_utils/cub/device/device_reduce.cuh"
#include "src/gpu_utils/cub/device/device_scan.cuh"
#include "src/gpu_utils/cub/device/device_select.cuh"
#include "src/gpu_utils/cub/iterator/transform_input_iterator.cuh"

template<typename T>
struct Square
{
    __host__ __device__ __forceinline__
    T operator()(const T& a) const {
        return a*a;
    }
};
//using namespace cub;

template <
    typename                T,
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    cub::BlockLoadAlgorithm     LALGORITHM,
    cub::BlockReduceAlgorithm    ALGORITHM>
__global__ void BlockSumKernel(
    T         *d_in,          // Tile of input
    T         *d_out,
    int       in_size)         // Tile aggregate
{
    // Specialize BlockReduce type for our thread block
    typedef cub::BlockLoad<T, BLOCK_THREADS, ITEMS_PER_THREAD, LALGORITHM> BlockLoadT;
    typedef cub::BlockReduce<T, BLOCK_THREADS, ALGORITHM> BlockReduceT;
    // Shared memory
    __shared__ union {
        typename BlockLoadT::TempStorage load;
        typename BlockReduceT::TempStorage reduce;
    } temp_storage;
    // Per-thread tile data
    T data[ITEMS_PER_THREAD];
    //cub::LoadDirectBlockedVectorized(threadIdx.x,
    //        d_in+blockIdx.x*BLOCK_THREADS*ITEMS_PER_THREAD, 
    //        data);
    int offset = blockIdx.x*BLOCK_THREADS*ITEMS_PER_THREAD;
    int valid_items = min(BLOCK_THREADS*ITEMS_PER_THREAD, in_size - offset);
    BlockLoadT(temp_storage.load).Load(d_in+offset, data, valid_items, T());

    for(int item=0; item<ITEMS_PER_THREAD; ++item)
        data[item] *= data[item];
    // Compute sum
    T aggregate = BlockReduceT(temp_storage.reduce).Sum(data);
    // Store aggregate and elapsed clocks
    if (threadIdx.x == 0)
    {
        //if(blockIdx.x % 1000000 == 0)
        //    printf("%f\n", aggregate);
        atomicAdd(d_out, aggregate);
    }
}

template <typename T>
static T getSquareSumOnBlock(CudaUnifedPtr<T> &ptr)
{
#ifdef DEBUG_CUDA
if (ptr.size == 0)
	printf("DEBUG_ERROR: getSquareSumOnDevice called with pointer of zero size.\n");
if (ptr.d_ptr == NULL)
	printf("DEBUG_ERROR: getSquareSumOnDevice called with null device pointer.\n");
if (ptr.getAllocator() == NULL && CustomAlloc)
	printf("DEBUG_ERROR: getSquareSumOnDevice called with null allocator.\n");
#endif
	CudaUnifedPtr<T>  val(ptr.getStream(), ptr.gpu_id, 1);
	val.alloc();

	int sumSize = (int) ceilf(((XFLOAT) ptr.getSize())/((XFLOAT)(512*4)));
    BlockSumKernel<T, 512, 4, cub::BLOCK_LOAD_VECTORIZE, cub::BLOCK_REDUCE_RAKING><<<sumSize, 512, 0, ptr.getStream()>>>(~ptr, ~val, ptr.getSize());
	ptr.streamSync();

	return val[0];
}


template <typename T, bool CustomAlloc>
static T getSquareSumOnBlock(CudaGlobalPtr<T, CustomAlloc> &ptr)
{
#ifdef DEBUG_CUDA
if (ptr.size == 0)
	printf("DEBUG_ERROR: getSquareSumOnDevice called with pointer of zero size.\n");
if (ptr.d_ptr == NULL)
	printf("DEBUG_ERROR: getSquareSumOnDevice called with null device pointer.\n");
if (ptr.getAllocator() == NULL && CustomAlloc)
	printf("DEBUG_ERROR: getSquareSumOnDevice called with null allocator.\n");
#endif
	CudaGlobalPtr<T, CustomAlloc>  val(1, ptr.getStream(), ptr.getAllocator());
	val.device_alloc();

	int sumSize = (int) ceilf(((XFLOAT) ptr.getSize())/((XFLOAT)(512*4)));
    BlockSumKernel<T, 512, 4, cub::BLOCK_LOAD_VECTORIZE, cub::BLOCK_REDUCE_RAKING><<<sumSize, 512, 0, ptr.getStream()>>>(~ptr, ~val, ptr.getSize());
	ptr.streamSync();
    val.cp_to_host();

	return val[0];
}

template <typename T, bool CustomAlloc>
static T getSquareSumOnDevice(CudaGlobalPtr<T, CustomAlloc> &ptr)
{
#ifdef DEBUG_CUDA
if (ptr.size == 0)
	printf("DEBUG_ERROR: getSquareSumOnDevice called with pointer of zero size.\n");
if (ptr.d_ptr == NULL)
	printf("DEBUG_ERROR: getSquareSumOnDevice called with null device pointer.\n");
if (ptr.getAllocator() == NULL && CustomAlloc)
	printf("DEBUG_ERROR: getSquareSumOnDevice called with null allocator.\n");
#endif
    cub::TransformInputIterator<T, Square<T>, T*> input_iter(~ptr, Square<T>());
	CudaGlobalPtr<T, CustomAlloc>  val(1, ptr.getStream(), ptr.getAllocator());
	val.device_alloc();
    val.device_init(T());
    void *d_temp_storage = NULL;
	size_t temp_storage_size = 0;

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_size, input_iter, ~val, ptr.size, ptr.getStream()));

    cudaMalloc(&d_temp_storage, temp_storage_size);

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_size, input_iter, ~val, ptr.size, ptr.getStream()));

	ptr.streamSync();
	val.cp_to_host();
    cudaFree(d_temp_storage);
	return val[0];
}

template <typename T>
static std::pair<int, T> getArgMaxOnDevice(CudaGlobalPtr<T> &ptr)
{
#ifdef DEBUG_CUDA
if (ptr.size == 0)
	printf("DEBUG_WARNING: getArgMaxOnDevice called with pointer of zero size.\n");
if (ptr.d_ptr == NULL)
	printf("DEBUG_WARNING: getArgMaxOnDevice called with null device pointer.\n");
if (ptr.getAllocator() == NULL)
	printf("DEBUG_WARNING: getArgMaxOnDevice called with null allocator.\n");
#endif
	CudaGlobalPtr<cub::KeyValuePair<int, T> >  max_pair(1, ptr.getStream(), ptr.getAllocator());
	max_pair.device_alloc();
	size_t temp_storage_size = 0;

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::ArgMax( NULL, temp_storage_size, ~ptr, ~max_pair, ptr.size));

	if(temp_storage_size==0)
		temp_storage_size=1;

	CudaCustomAllocator::Alloc* alloc = ptr.getAllocator()->alloc(temp_storage_size);

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::ArgMax( alloc->getPtr(), temp_storage_size, ~ptr, ~max_pair, ptr.size, ptr.getStream()));

	max_pair.cp_to_host();
	ptr.streamSync();

	ptr.getAllocator()->free(alloc);

	std::pair<int, T> pair;
	pair.first = max_pair[0].key;
	pair.second = max_pair[0].value;

	return pair;
}

template <typename T>
static std::pair<int, T> getArgMinOnDevice(CudaGlobalPtr<T> &ptr)
{
#ifdef DEBUG_CUDA
if (ptr.size == 0)
	printf("DEBUG_WARNING: getArgMinOnDevice called with pointer of zero size.\n");
if (ptr.d_ptr == NULL)
	printf("DEBUG_WARNING: getArgMinOnDevice called with null device pointer.\n");
if (ptr.getAllocator() == NULL)
	printf("DEBUG_WARNING: getArgMinOnDevice called with null allocator.\n");
#endif
	CudaGlobalPtr<cub::KeyValuePair<int, T> >  min_pair(1, ptr.getStream(), ptr.getAllocator());
	min_pair.device_alloc();
	size_t temp_storage_size = 0;

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::ArgMin( NULL, temp_storage_size, ~ptr, ~min_pair, ptr.size));

	if(temp_storage_size==0)
		temp_storage_size=1;

	CudaCustomAllocator::Alloc* alloc = ptr.getAllocator()->alloc(temp_storage_size);

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::ArgMin( alloc->getPtr(), temp_storage_size, ~ptr, ~min_pair, ptr.size, ptr.getStream()));

	min_pair.cp_to_host();
	ptr.streamSync();

	ptr.getAllocator()->free(alloc);

	std::pair<int, T> pair;
	pair.first = min_pair[0].key;
	pair.second = min_pair[0].value;

	return pair;
}

template <typename T>
struct CustomMin
{
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        if(b >= 1.e-9) return (b < a) ? b : a;
        else return a;
    }
};

template <typename T>
static T reduceOnDevice(CudaUnifedPtr<T> &ptr)
{
#ifdef DEBUG_CUDA
if (ptr.size == 0)
	printf("DEBUG_ERROR: getMaxOnDevice called with pointer of zero size.\n");
if (ptr.ptr == NULL)
	printf("DEBUG_ERROR: getMaxOnDevice called with null device pointer.\n");
if (ptr.getAllocator() == NULL)
	printf("DEBUG_ERROR: getMaxOnDevice called with null allocator.\n");
#endif
	CudaUnifedPtr<T>  reduce_val(ptr.getStream(), ptr.gpu_id, 1);
	reduce_val.alloc();
	size_t temp_storage_size = 0;
    CustomMin<T> myOp;
    T init = std::numeric_limits<T>::max();

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::Reduce( NULL, temp_storage_size, ~ptr, ~reduce_val, ptr.size, myOp, init, ptr.getStream()));
    void *d_temp_storage = NULL;

	if(temp_storage_size==0)
		temp_storage_size=1;

    //if(CustomAlloc) {
    //    CudaCustomAllocator::Alloc* alloc = ptr.getAllocator()->alloc(temp_storage_size);

    //    DEBUG_HANDLE_ERROR(cub::DeviceReduce::Max( alloc->getPtr(), temp_storage_size, ~ptr, ~reduce_val, ptr.size, ptr.getStream()));
    //} else {
    //}
    DEBUG_HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_size));
    DEBUG_HANDLE_ERROR(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_size, ~ptr, ~reduce_val, ptr.size, myOp, init, ptr.getStream()));

	ptr.streamSync();

    cudaFree(d_temp_storage);

    //if(CustomAlloc)
	//    ptr.getAllocator()->free(alloc);

	return reduce_val[0];
}

template <typename T, bool CustomAlloc = true>
static T reduceOnDevice(CudaGlobalPtr<T, CustomAlloc > &ptr)
{
#ifdef DEBUG_CUDA
if (ptr.size == 0)
	printf("DEBUG_ERROR: getMaxOnDevice called with pointer of zero size.\n");
if (ptr.d_ptr == NULL)
	printf("DEBUG_ERROR: getMaxOnDevice called with null device pointer.\n");
if (ptr.getAllocator() == NULL)
	printf("DEBUG_ERROR: getMaxOnDevice called with null allocator.\n");
#endif
	CudaGlobalPtr<T, CustomAlloc >  reduce_val(1, ptr.getStream(), ptr.getAllocator());
	reduce_val.device_alloc();
	size_t temp_storage_size = 0;
    CustomMin<T> myOp;
    T init = std::numeric_limits<T>::max();

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::Reduce( NULL, temp_storage_size, ~ptr, ~reduce_val, ptr.size, myOp, init, ptr.getStream()));
    void *d_temp_storage = NULL;

	if(temp_storage_size==0)
		temp_storage_size=1;

    //if(CustomAlloc) {
    //    CudaCustomAllocator::Alloc* alloc = ptr.getAllocator()->alloc(temp_storage_size);

    //    DEBUG_HANDLE_ERROR(cub::DeviceReduce::Max( alloc->getPtr(), temp_storage_size, ~ptr, ~reduce_val, ptr.size, ptr.getStream()));
    //} else {
    //}
    DEBUG_HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_size));
    DEBUG_HANDLE_ERROR(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_size, ~ptr, ~reduce_val, ptr.size, myOp, init, ptr.getStream()));

	ptr.streamSync();
	reduce_val.cp_to_host();

    cudaFree(d_temp_storage);

    //if(CustomAlloc)
	//    ptr.getAllocator()->free(alloc);

	return reduce_val[0];
}

template <typename T>
static T getMaxOnDevice(CudaUnifedPtr<T> &ptr)
{
#ifdef DEBUG_CUDA
if (ptr.size == 0)
	printf("DEBUG_ERROR: getMaxOnDevice called with pointer of zero size.\n");
if (ptr.d_ptr == NULL)
	printf("DEBUG_ERROR: getMaxOnDevice called with null device pointer.\n");
if (ptr.getAllocator() == NULL)
	printf("DEBUG_ERROR: getMaxOnDevice called with null allocator.\n");
#endif
	CudaUnifedPtr<T>  max_val(ptr.getStream(), ptr.gpu_id, 1);
	max_val.alloc();
	size_t temp_storage_size = 0;

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::Max( NULL, temp_storage_size, ~ptr, ~max_val, ptr.size, ptr.getStream()));
    void *d_temp_storage = NULL;

	if(temp_storage_size==0)
		temp_storage_size=1;

    //if(CustomAlloc) {
    //    CudaCustomAllocator::Alloc* alloc = ptr.getAllocator()->alloc(temp_storage_size);

    //    DEBUG_HANDLE_ERROR(cub::DeviceReduce::Max( alloc->getPtr(), temp_storage_size, ~ptr, ~max_val, ptr.size, ptr.getStream()));
    //} else {
    //}
    DEBUG_HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_size));
    //std::cout << temp_storage_size << " " << d_temp_storage << std::endl;
    std::cout << ~ptr << " " << ptr.size << " " << ~max_val << " " << ptr.getStream() << std::endl;
    DEBUG_HANDLE_ERROR(cub::DeviceReduce::Max(d_temp_storage, temp_storage_size, ~ptr, ~max_val, ptr.size, ptr.getStream()));

	ptr.streamSync();

    cudaFree(d_temp_storage);

    //if(CustomAlloc)
	//    ptr.getAllocator()->free(alloc);

	return max_val[0];
}

template <typename T, bool CustomAlloc = true>
static T getMaxOnDevice(CudaGlobalPtr<T, CustomAlloc > &ptr)
{
#ifdef DEBUG_CUDA
if (ptr.size == 0)
	printf("DEBUG_ERROR: getMaxOnDevice called with pointer of zero size.\n");
if (ptr.d_ptr == NULL)
	printf("DEBUG_ERROR: getMaxOnDevice called with null device pointer.\n");
if (ptr.getAllocator() == NULL)
	printf("DEBUG_ERROR: getMaxOnDevice called with null allocator.\n");
#endif
	CudaGlobalPtr<T, CustomAlloc >  max_val(1, ptr.getStream(), ptr.getAllocator());
	max_val.device_alloc();
	size_t temp_storage_size = 0;

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::Max( NULL, temp_storage_size, ~ptr, ~max_val, ptr.size, ptr.getStream()));
    void *d_temp_storage = NULL;

	if(temp_storage_size==0)
		temp_storage_size=1;

    //if(CustomAlloc) {
    //    CudaCustomAllocator::Alloc* alloc = ptr.getAllocator()->alloc(temp_storage_size);

    //    DEBUG_HANDLE_ERROR(cub::DeviceReduce::Max( alloc->getPtr(), temp_storage_size, ~ptr, ~max_val, ptr.size, ptr.getStream()));
    //} else {
    //}
    DEBUG_HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_size));
    //std::cout << temp_storage_size << " " << d_temp_storage << std::endl;
    DEBUG_HANDLE_ERROR(cub::DeviceReduce::Max(d_temp_storage, temp_storage_size, ~ptr, ~max_val, ptr.size, ptr.getStream()));

	max_val.cp_to_host();
	ptr.streamSync();

    cudaFree(d_temp_storage);

    //if(CustomAlloc)
	//    ptr.getAllocator()->free(alloc);

	return max_val[0];
}

template <typename T>
static T getMinOnDevice(CudaGlobalPtr<T> &ptr)
{
#ifdef DEBUG_CUDA
if (ptr.size == 0)
	printf("DEBUG_ERROR: getMinOnDevice called with pointer of zero size.\n");
if (ptr.d_ptr == NULL)
	printf("DEBUG_ERROR: getMinOnDevice called with null device pointer.\n");
if (ptr.getAllocator() == NULL)
	printf("DEBUG_ERROR: getMinOnDevice called with null allocator.\n");
#endif
	CudaGlobalPtr<T >  min_val(1, ptr.getStream(), ptr.getAllocator());
	min_val.device_alloc();
	size_t temp_storage_size = 0;

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::Min( NULL, temp_storage_size, ~ptr, ~min_val, ptr.size));

	if(temp_storage_size==0)
		temp_storage_size=1;

	CudaCustomAllocator::Alloc* alloc = ptr.getAllocator()->alloc(temp_storage_size);

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::Min( alloc->getPtr(), temp_storage_size, ~ptr, ~min_val, ptr.size, ptr.getStream()));

	min_val.cp_to_host();
	ptr.streamSync();

	ptr.getAllocator()->free(alloc);

	return min_val[0];
}

template <typename T>
static T getSumOnDevice(CudaGlobalPtr<T> &ptr)
{
#ifdef DEBUG_CUDA
if (ptr.size == 0)
	printf("DEBUG_ERROR: getSumOnDevice called with pointer of zero size.\n");
if (ptr.d_ptr == NULL)
	printf("DEBUG_ERROR: getSumOnDevice called with null device pointer.\n");
if (ptr.getAllocator() == NULL)
	printf("DEBUG_ERROR: getSumOnDevice called with null allocator.\n");
#endif
	CudaGlobalPtr<T >  val(1, ptr.getStream(), ptr.getAllocator());
	val.device_alloc();
	size_t temp_storage_size = 0;

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::Sum( NULL, temp_storage_size, ~ptr, ~val, ptr.size));

	if(temp_storage_size==0)
		temp_storage_size=1;

	CudaCustomAllocator::Alloc* alloc = ptr.getAllocator()->alloc(temp_storage_size);

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::Sum( alloc->getPtr(), temp_storage_size, ~ptr, ~val, ptr.size, ptr.getStream()));

	val.cp_to_host();
	ptr.streamSync();

	ptr.getAllocator()->free(alloc);

	return val[0];
}

template <typename T>
static void sortOnDevice(CudaGlobalPtr<T> &in, CudaGlobalPtr<T> &out)
{
#ifdef DEBUG_CUDA
if (in.size == 0 || out.size == 0)
	printf("DEBUG_ERROR: sortOnDevice called with pointer of zero size.\n");
if (in.d_ptr == NULL || out.d_ptr == NULL)
	printf("DEBUG_ERROR: sortOnDevice called with null device pointer.\n");
if (in.getAllocator() == NULL)
	printf("DEBUG_ERROR: sortOnDevice called with null allocator.\n");
#endif
	size_t temp_storage_size = 0;

	cudaStream_t stream = in.getStream();

	DEBUG_HANDLE_ERROR(cub::DeviceRadixSort::SortKeys( NULL, temp_storage_size, ~in, ~out, in.size));

	if(temp_storage_size==0)
		temp_storage_size=1;

	CudaCustomAllocator::Alloc* alloc = in.getAllocator()->alloc(temp_storage_size);

	DEBUG_HANDLE_ERROR(cub::DeviceRadixSort::SortKeys( alloc->getPtr(), temp_storage_size, ~in, ~out, in.size, 0, sizeof(T) * 8, stream));

	alloc->markReadyEvent(stream);
	alloc->doFreeWhenReady();
}

template <typename T>
static void sortDescendingOnDevice(CudaGlobalPtr<T> &in, CudaGlobalPtr<T> &out)
{
#ifdef DEBUG_CUDA
if (in.size == 0 || out.size == 0)
	printf("DEBUG_ERROR: sortDescendingOnDevice called with pointer of zero size.\n");
if (in.d_ptr == NULL || out.d_ptr == NULL)
	printf("DEBUG_ERROR: sortDescendingOnDevice called with null device pointer.\n");
if (in.getAllocator() == NULL)
	printf("DEBUG_ERROR: sortDescendingOnDevice called with null allocator.\n");
#endif
	size_t temp_storage_size = 0;

	cudaStream_t stream = in.getStream();

	DEBUG_HANDLE_ERROR(cub::DeviceRadixSort::SortKeysDescending( NULL, temp_storage_size, ~in, ~out, in.size));

	if(temp_storage_size==0)
		temp_storage_size=1;

	CudaCustomAllocator::Alloc* alloc = in.getAllocator()->alloc(temp_storage_size);

	DEBUG_HANDLE_ERROR(cub::DeviceRadixSort::SortKeysDescending( alloc->getPtr(), temp_storage_size, ~in, ~out, in.size, 0, sizeof(T) * 8, stream));

	alloc->markReadyEvent(stream);
	alloc->doFreeWhenReady();

}

class AllocatorThrustWrapper
{
public:
    // just allocate bytes
    typedef char value_type;
	std::vector<CudaCustomAllocator::Alloc*> allocs;
	CudaCustomAllocator *allocator;

    AllocatorThrustWrapper(CudaCustomAllocator *allocator):
		allocator(allocator)
	{}

    ~AllocatorThrustWrapper()
    {
    	for (int i = 0; i < allocs.size(); i ++)
    		allocator->free(allocs[i]);
    }

    char* allocate(std::ptrdiff_t num_bytes)
    {
    	CudaCustomAllocator::Alloc* alloc = allocator->alloc(num_bytes);
    	allocs.push_back(alloc);
    	return (char*) alloc->getPtr();
    }

    void deallocate(char* ptr, size_t n)
    {
    	//TODO fix this (works fine without it though) /Dari
    }
};

template <typename T>
struct MoreThanCubOpt
{
	T compare;
	MoreThanCubOpt(T compare) : compare(compare) {}
	__device__ __forceinline__
	bool operator()(const T &a) const {
		return (a > compare);
	}
};

template <typename T>
struct IsNanOp
{
	__device__ __forceinline__
	bool operator()(const T &a) const {
		return isnan(a);
	}
};

template <typename T, typename SelectOp>
static int filterOnDevice(CudaGlobalPtr<T> &in, CudaGlobalPtr<T> &out, SelectOp select_op)
{
#ifdef DEBUG_CUDA
if (in.size == 0 || out.size == 0)
	printf("DEBUG_ERROR: filterOnDevice called with pointer of zero size.\n");
if (in.d_ptr == NULL || out.d_ptr == NULL)
	printf("DEBUG_ERROR: filterOnDevice called with null device pointer.\n");
if (in.getAllocator() == NULL)
	printf("DEBUG_ERROR: filterOnDevice called with null allocator.\n");
#endif
	size_t temp_storage_size = 0;

	cudaStream_t stream = in.getStream();

	CudaGlobalPtr<int>  num_selected_out(1, stream, in.getAllocator());
	num_selected_out.device_alloc();

	DEBUG_HANDLE_ERROR(cub::DeviceSelect::If(NULL, temp_storage_size, ~in, ~out, ~num_selected_out, in.size, select_op, stream));

	if(temp_storage_size==0)
		temp_storage_size=1;

	CudaCustomAllocator::Alloc* alloc = in.getAllocator()->alloc(temp_storage_size);

	DEBUG_HANDLE_ERROR(cub::DeviceSelect::If(alloc->getPtr(), temp_storage_size, ~in, ~out, ~num_selected_out, in.size, select_op, stream));

	num_selected_out.cp_to_host();
	DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));

	in.getAllocator()->free(alloc);
	return num_selected_out[0];
}

template <typename T>
static void scanOnDevice(CudaGlobalPtr<T> &in, CudaGlobalPtr<T> &out)
{
#ifdef DEBUG_CUDA
if (in.size == 0 || out.size == 0)
	printf("DEBUG_ERROR: scanOnDevice called with pointer of zero size.\n");
if (in.d_ptr == NULL || out.d_ptr == NULL)
	printf("DEBUG_ERROR: scanOnDevice called with null device pointer.\n");
if (in.getAllocator() == NULL)
	printf("DEBUG_ERROR: scanOnDevice called with null allocator.\n");
#endif
	size_t temp_storage_size = 0;

	cudaStream_t stream = in.getStream();

	DEBUG_HANDLE_ERROR(cub::DeviceScan::InclusiveSum( NULL, temp_storage_size, ~in, ~out, in.size));

	if(temp_storage_size==0)
		temp_storage_size=1;

	CudaCustomAllocator::Alloc* alloc = in.getAllocator()->alloc(temp_storage_size);

	DEBUG_HANDLE_ERROR(cub::DeviceScan::InclusiveSum( alloc->getPtr(), temp_storage_size, ~in, ~out, in.size, stream));

	alloc->markReadyEvent(stream);
	alloc->doFreeWhenReady();
}

#endif
