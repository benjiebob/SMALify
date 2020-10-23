If you are having problems installing PyTorch3D on Windows with PyTorch 1.6, try the following fixes:

1. In file venv/Lib/site-packages/torch/include/pybind11/cast.h, change:
    * (l1449)` explicit operator type&() { return*(this->value); }`
    * to `explicit operator type&() { return *((type*)this->value); }`

2. In file venv\Lib\site-packages\torch\include\torch\csrc\jit\runtime, change
    * (l160) `static constexpr size_t DEPTH_LIMIT = 128;`
    * to `static const size_t DEPTH_LIMIT = 128;`
    
3. In file venv\Lib\site-packages\torch\include\torch\csrc\jit\api, change:
    * all instances of `CONSTEXPR_EXCEPT_WIN_CUDA `
    * to `const`