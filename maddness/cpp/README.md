# MADDness C++ code

```bash
source halut.env
./build.sh
```

## GCC build

```bash
g++-9.2.0 test/quantize/test_mithral.cpp -I/home/msc22f5/Documents/bolt/cpp/src/utils/ -I/home/msc22f5/Documents/bolt/cpp/src/external/eigen -I/home/msc22f5/Documents/bolt/cpp/src/quantize/ -I/home/msc22f5/Documents/bolt/cpp/quantize/ -I/home/msc22f5/Documents/bolt/cpp/test/testing_utils/ -O3 -march=native -mavx -ffast-math -std=c++14 -o maddness
```
## Clang build

```bash
clang++-12.0.1 test/quantize/test_mithral.cpp -I/home/msc22f5/Documents/bolt/cpp/src/utils/ -I/home/msc22f5/Documents/bolt/cpp/src/external/eigen -I/home/msc22f5/Documents/bolt/cpp/src/quantize/ -I/home/msc22f5/Documents/bolt/cpp/quantize/ -I/home/msc22f5/Documents/bolt/cpp/test/testing_utils/ -O3 -march=native -mavx -ffast-math -std=c++14 -o maddness -stdlib=libc++
```