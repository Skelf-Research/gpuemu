You can get surprisingly far on Linux without a physical GPU, but you have to be clear about what you’re simulating:

* **Correctness / functional bring-up** (kernels compile, run, produce correct outputs)
* **API integration** (CUDA/HIP/OpenCL/Vulkan code paths work end-to-end)
* **Performance** (occupancy, memory coalescing, bandwidth, latency) — this is the one you *can’t* truly simulate without real hardware, but you can approximate and catch big mistakes.

Here are the best options, grouped by what they’re good for.

## 1) “CPU device” backends (best for functional correctness)

### OpenCL: POCL (Portable Computing Language)

* Runs OpenCL kernels on CPU.
* Great for validating OpenCL kernel logic, memory indexing, barriers, etc.
* Use it to test your OpenCL host code too (queueing, buffers, events).

### SYCL: DPC++ “CPU device” + oneAPI runtime

* Lets you run SYCL kernels on CPU as a backend.
* Useful if your stack is SYCL-first.

### Vulkan compute: Lavapipe (Mesa software Vulkan)

* Vulkan driver implemented in software (CPU).
* Can run compute pipelines without a GPU.
* Great for exercising Vulkan SPIR-V + descriptor/buffer plumbing.

**When this wins:** you want CI to run kernels and you want deterministic, debuggable execution.

---

## 2) Vendor simulators / emulators (closest to “GPU semantics,” still limited)

### NVIDIA CUDA: device emulation isn’t like the old days

* Historically CUDA had “device emulation,” but modern CUDA largely expects real GPUs.
* You *can* still do a lot via:

  * CPU builds + heavy unit tests of kernel logic extracted to scalar code
  * Sanitizers (see below) and compilation checks

### AMD HIP/ROCm

* ROCm generally assumes AMD GPUs present for actual execution.
* But you can still compile, run host-side tests, and use CPU fallbacks for some math libraries—just don’t expect true kernel execution without hardware.

**When this wins:** you’re targeting a vendor stack and want compile-time + tooling validation, but not full execution.

---

## 3) Software renderers for graphics APIs (useful for compute too sometimes)

If your “GPU dev” is **graphics/compute via OpenGL/Vulkan**, Mesa’s software stacks are a real workhorse:

* **llvmpipe**: software OpenGL (CPU)
* **lavapipe**: software Vulkan (CPU)

This lets you run shader compilation, SPIR-V validation, pipeline layout correctness, descriptor binding correctness, etc.

---

## 4) Tooling to catch GPU-style bugs without a GPU (high leverage)

Even without execution on real hardware, you can catch a lot:

### SPIR-V toolchain

* `glslangValidator` / `spirv-as` / `spirv-val`
* Catches invalid SPIR-V, type/layout issues, descriptor mistakes.

### Shader/kernel static analysis

* SPIR-V validation + “bounds style” checks via your own asserts/instrumentation.

### “Sanitizer style” for kernels

* On CPU backends (POCL, SYCL CPU, lavapipe), you can run with:

  * AddressSanitizer / UBSan / ThreadSanitizer on the host program
  * And get much better crash traces than on real GPUs.

**Key trick:** build kernels so they can compile into a CPU reference path (same indexing math), and run that in CI with sanitizers.

---

## 5) QEMU / virtualization isn’t the answer for GPU compute

QEMU can emulate *CPUs* well; it doesn’t meaningfully emulate modern CUDA/ROCm GPUs for compute. PCI passthrough needs a real GPU.

---

## 6) Practical workflow that works well in teams (no-GPU dev + GPU CI)

**Local (no GPU):**

1. Compile kernels every build (ensures toolchain correctness).
2. Run kernels on CPU backend:

   * OpenCL → POCL
   * Vulkan compute → lavapipe
   * SYCL → CPU device
3. Maintain a **reference CPU implementation** of each kernel and compare outputs.
4. Use sanitizers + property tests (randomized shapes/strides).

**CI (with GPU, even a single cheap one):**

* Run nightly or on merge:

  * performance smoke tests
  * real GPU correctness
  * architecture-specific behavior

If you truly have *zero GPUs anywhere*, then you’re essentially limited to “functional + API correctness,” not performance validation.

---

## 7) If you tell me your target stack, I’ll give you a concrete setup

Answering this changes the recommendation a lot:

* Are you writing **CUDA**, **HIP/ROCm**, **OpenCL**, **SYCL**, or **Vulkan compute**?
* Do you need **graphics + compute**, or compute-only?
* Are you trying to support multiple vendors?

If you reply with your stack (even just “OpenCL” or “CUDA”), I’ll give you:

* the exact packages to install on Ubuntu/Debian
* how to force the CPU backend
* a minimal “vector add” test harness you can use as a CI gate

