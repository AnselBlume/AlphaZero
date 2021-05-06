from setuptools import setup, Extension

# Compile *rollout.cpp* into a shared library
# See https://stackoverflow.com/questions/33738885/python-setuptools-not-including-c-standard-library-headers
# and https://nesi.github.io/perf-training/python-scatter/ctypes for examples
setup(
    #...
    ext_modules=[
        Extension(
            'rollout', # Name of the library
            ['rollout.cpp', 'thc.cpp'],
            extra_compile_args=[
                '-std=c++20',
                'w', # Suppress warnings
                '-O3'
            ])
    ],
)
