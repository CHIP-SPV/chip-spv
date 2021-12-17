# CHIP-SPV

CHIP-SPV is a HIP implementation that abstracts HIP API, providing a
set of base classes that can be derived from to implement an
additional, SPIR-V capable backend on which to execute HIP
calls.

Currently CHIP-SPV supports OpenCL and Level Zero as backend alternatives.

This project is an integration of [HIPCL](https://github.com/cpc/hipcl) and
[HIPLZ](https://github.com/jz10/anl-gt-gpu/) projects.

## Prerequisites

* Cmake >= 3.18.0
* Clang with SPIR-V patches: [hipcl-clang](https://github.com/parmance/llvm-project)
* SPIRV-LLVM-Translator: [llvm-spirv](https://github.com/KhronosGroup/SPIRV-LLVM-Translator)
* For Level Zero Backend
  * [Intel Compute Runtime](https://github.com/intel/compute-runtime)
* For OpenCL Backend
  * An OpenCL implementation with (at least partial) 2.x support;
    HIPCL requires Shared Virtual Memory and clCreateProgramWithIL()
    support

## Downloading and Building Clang

Downloading:

```bash
git clone git@github.com:parmance/llvm-project.git -b hip2spirv-v5
cd llvm-project/llvm/projects
git clone git@github.com:KhronosGroup/SPIRV-LLVM-Translator.git
cd SPIRV-LLVM-Translator
git checkout 8679b960f46a5095e4230e1e350cef035f6f6b9e
```

Building:

```bash
cd llvm-project/llvm
mkdir build
cd build
cmake .. -DLLVM_ENABLE_PROJECTS="clang" \
  -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_DIR}
make
make install
```

## Downloading Sources

```bash
git clone https://github.com/CHIP-SPV/chip-spv.git
cd chip-spv
git submodule update --init --recursive
```

## Building

```bash
# export PATH=${LLVM_INSTALL_DIR}/bin:$PATH
mkdir build
cd build

cmake .. -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS=-O2
make
```


Note: -O2 CXX optimization flag is currently required.
See [Issue #2](https://github.com/CHIP-SPV/chip-spv/issues/2).

## Testing

Run tests on the default backend:

```bash
make test
```

Run tests on a specific backend:

```bash
CHIP_BE=<backend> make test
```

Where the `backend` is a backend identification. Possible values for it are
`level0` and `opencl`.
>>>>>>> 90d48a9... Add mention of the -O2 requirement to README.md
