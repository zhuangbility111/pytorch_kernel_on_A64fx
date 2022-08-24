from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

compile_extra_args = ['-O3', '-Kopenmp', '-Nlibomp', '-Kfast']
link_extra_args = [ '-Kopenmp', '-Nlibomp', '-Kfast', '-Kzfill']

setup(
    name='spmm_utils',
    ext_modules=[
        CppExtension(
            name='spmm_utils',
            sources=['spmm_for_transposed.cpp'],
            language='c++',
            extra_compile_args = compile_extra_args,
            extra_link_args = link_extra_args)],
    cmdclass={
        'build_ext': BuildExtension
    }
)
