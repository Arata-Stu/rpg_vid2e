from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": ["-O3", "--use_fast_math"],
}

setup(
    name='esim_torch',
    package_dir={'':'src'},
    packages=['esim_torch'],
    ext_modules=[
        CUDAExtension(name='esim_cuda',
                      sources=[
                      'src/esim_torch/esim_cuda_kernel.cu',
                      ],
                      extra_compile_args=extra_compile_args
                     )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
