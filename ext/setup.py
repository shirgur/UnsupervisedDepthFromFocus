from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gauss_psf_cuda',
    ext_modules=[
        CUDAExtension('gauss_psf_cuda', [
            'gauss_psf_cuda.cpp',
            'gauss_psf_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })