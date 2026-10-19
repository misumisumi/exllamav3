#include "cuda_host.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"

// The runtime records a failed call's error as the last error as well. Errors these functions report (or deliberately
// ignore) must be consumed, or the next cudaGetLastError() elsewhere (torch's launch checks) attributes them to
// something unrelated
static void consume_error(cudaError_t cr)
{
    if (cr != cudaSuccess) cudaGetLastError();
}

void cuda_host_register(uintptr_t ptr, size_t nbytes, unsigned int flags)
{
    cudaError_t cr = cudaHostRegister(reinterpret_cast<void*>(ptr), nbytes, flags);

    // A region can only be pinned once per process, but ranks sharing an arena may each try to
    // register it
    consume_error(cr);
    if (cr == cudaErrorHostMemoryAlreadyRegistered) return;

    TORCH_CHECK(
        cr == cudaSuccess,
        "cudaHostRegister(", reinterpret_cast<void*>(ptr), ", ", nbytes, ") failed: ",
        cudaGetErrorString(cr)
    );
}

void cuda_host_unregister(uintptr_t ptr)
{
    cudaError_t cr = cudaHostUnregister(reinterpret_cast<void*>(ptr));

    // Teardown is racy by nature: the region may already have been released, or the runtime may
    // be unloading while shared segments are still being torn down. Both are benign here
    consume_error(cr);
    if (cr == cudaErrorHostMemoryNotRegistered || cr == cudaErrorCudartUnloading) return;

    TORCH_CHECK(
        cr == cudaSuccess,
        "cudaHostUnregister(", reinterpret_cast<void*>(ptr), ") failed: ", cudaGetErrorString(cr)
    );
}

uintptr_t cuda_host_get_device_pointer(uintptr_t ptr)
{
    // Device-side alias of memory registered with cudaHostRegisterMapped. Equal to the host
    // pointer under UVA (Linux desktop), but distinct under WDDM (native Windows, and potentially
    // WSL2), where the host pointer is not usable in kernels and this alias must be passed instead
    void* dev_ptr = nullptr;
    cudaError_t cr = cudaHostGetDevicePointer(&dev_ptr, reinterpret_cast<void*>(ptr), 0);
    consume_error(cr);

    TORCH_CHECK(
        cr == cudaSuccess,
        "cudaHostGetDevicePointer(", reinterpret_cast<void*>(ptr), ") failed: ",
        cudaGetErrorString(cr)
    );

    return reinterpret_cast<uintptr_t>(dev_ptr);
}

int cuda_device_get_attribute(int attr, int device)
{
    int value = 0;
#if defined(USE_ROCM)
    // The caller passes CUDA's cudaDeviceAttr numbering (model_tp_cuda.py), which HIP's attribute enum does
    // not share: translate the values in use rather than query whatever attribute HIP has at that number
    TORCH_CHECK(attr == 91, "cuda_device_get_attribute: CUDA attribute ", attr, " has no HIP mapping here");
    cudaError_t cr = cudaDeviceGetAttribute(&value, cudaDevAttrCanUseHostPointerForRegisteredMem, device);
#else
    cudaError_t cr = cudaDeviceGetAttribute(&value, static_cast<cudaDeviceAttr>(attr), device);
#endif
    consume_error(cr);

    TORCH_CHECK(
        cr == cudaSuccess,
        "cudaDeviceGetAttribute(", attr, ", ", device, ") failed: ", cudaGetErrorString(cr)
    );

    return value;
}

at::Tensor pinned_cuda_view(const at::Tensor& t, int64_t device)
{
    // CUDA-device alias of a pinned host tensor: same storage, but with a device dtype/layout
    // so torch ops (matmul/cuBLAS) and extension kernels accept it directly, reading over PCIe
    // (zero-copy). The view does NOT own the memory; the caller must keep the pinned source
    // tensor alive for the alias's lifetime. Uses cudaHostGetDevicePointer rather than assuming
    // UVA pointer equality so the alias also holds under WDDM
    TORCH_CHECK(t.device().is_cpu(), "pinned_cuda_view: tensor must be a CPU tensor");
    TORCH_CHECK(t.is_pinned(), "pinned_cuda_view: tensor must be pinned");
    void* dev_ptr = nullptr;
    cudaError_t cr = cudaHostGetDevicePointer(&dev_ptr, t.data_ptr(), 0);
    consume_error(cr);
    TORCH_CHECK(
        cr == cudaSuccess,
        "cudaHostGetDevicePointer(", t.data_ptr(), ") failed: ", cudaGetErrorString(cr)
    );
    auto options = t.options().device(at::kCUDA, static_cast<c10::DeviceIndex>(device));
    return at::from_blob(dev_ptr, t.sizes(), t.strides(), [](void*) {}, options);
}

at::Tensor ats_cuda_view(const at::Tensor& t, int64_t device)
{
    // CUDA-device alias of pageable host memory on ATS systems (Grace/GB10), where the GPU
    // shares the process page tables: the device pointer is the host address, nothing is
    // pinned, and file-backed pages stay reclaimable page cache. The alias holds a reference
    // to the source tensor, so the mapping lives as long as any view of it. Built from a Storage
    // whose DataPtr names the device: at::from_blob asks the CUDA runtime which device owns the
    // pointer, and a pageable mapping is unknown to it
    TORCH_CHECK(t.device().is_cpu(), "ats_cuda_view: tensor must be a CPU tensor");
    TORCH_CHECK(t.is_contiguous(), "ats_cuda_view: tensor must be contiguous");
    c10::Device dev(c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(device));
    auto* keep = new at::Tensor(t);
    c10::DataPtr data_ptr(t.data_ptr(), keep, [](void* ctx) { delete static_cast<at::Tensor*>(ctx); }, dev);
    at::Storage storage(at::Storage::use_byte_size_t(), t.nbytes(), std::move(data_ptr), nullptr, false);
    at::Tensor out = at::empty({0}, t.options().device(dev));
    out.set_(storage, 0, t.sizes(), t.strides());
    return out;
}
