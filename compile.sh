g++ -o d2s_net d2s_net.cc -O3 -march=native -DMKL_LP64 -m64 -I/opt/intel/oneapi/mkl/2021.1.1//include  -Wl,--start-group /opt/intel/oneapi/mkl/2021.1.1//lib/intel64/libmkl_intel_lp64.a /opt/intel/oneapi/mkl/2021.1.1//lib/intel64/libmkl_sequential.a /opt/intel/oneapi/mkl/2021.1.1//lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl -I ../libxsmm/include -DNDEBUG --std=c++17
