/* Copyright 2023 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "nanobind/nanobind.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"

namespace jax::cuda {
namespace {

namespace nb = nanobind;
using MosaicHostFunc = void(gpuStream_t, void**);

void MosaicKernelCall(gpuStream_t stream, void** buffers, const char* opaque,
                      size_t opaque_len, XlaCustomCallStatus* status) {
  reinterpret_cast<MosaicHostFunc*>(opaque)(stream, buffers);
}

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("mosaic_gpu", MosaicKernelCall,
                                         "CUDA");

NB_MODULE(_mosaic_gpu, m) {
  // For now we don't expose anything, but loading this module has a side-effect
  // of registering the custom call.
}

}  // namespace
}  // namespace jax::cuda


