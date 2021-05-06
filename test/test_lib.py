# Refer here for this code: https://nesi.github.io/perf-training/python-scatter/ctypes
# Can also skip the build directory and glob by using g++ -o mylib.so -shared rollout.cpp thc.cpp
# as done here
import ctypes
import glob

# find the shared library, the path depends on the platform and Python version
libfile = glob.glob('build/*/*.so')[0]

# 1. open the shared library
mylib = ctypes.CDLL(libfile)

# 2. tell Python the argument and result types of function mysum
mylib.rollout.restype = ctypes.c_int
mylib.rollout.argtypes = [ctypes.c_char_p]

fen = 'k7/1Q6/1K6/8/8/8/8/8 b - - 0 1'.encode('ascii')
value = mylib.rollout(fen)

print(f'Value should be -1: {value}')
