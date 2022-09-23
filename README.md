# The Salary Man Problem on GPU

## Prerequisites

- `C` compilers
- OpenCL SDK for at least one of your platforms

## Build

1. Check the `CMakeLists.txt` to suits your environment variables and compiler
2. Setup the toolchain for your compiler
3. Run `cmake` to generate the build files
4. Run `cmake --build .\build --target main` to build the project

> An example of building with `ninja` on Windows
>
> ```shell
> cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=ninja -G Ninja -S .\ -B .\build
> cmake --build .\build --target main -j 3
> ```
