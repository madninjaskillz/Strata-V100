W=/work/strata-v100opt
cd $W && cmake -G Ninja -B build-opt -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_EXPERIMENTAL_SM60=ON -DSTRATA_GGML_DIR=/work/strata/third_party/llama.cpp > build-opt.log 2>&1 && cmake --build build-opt --target strata strata_kq -j 24 >> build-opt.log 2>&1
echo BUILD_EXIT=$? >> build-opt.log
