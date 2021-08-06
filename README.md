# datatools
Various scripts, libraries and tools for working with data.

## Prerequisites: Creating an Anaconda Environment (Python)
I'm using [Anaconda](https://www.anaconda.com/) for managing my Python environment. Follow the instructions [here](https://docs.anaconda.com/anaconda/install/index.html) to install Anaconda on your system, and then:
```bash
datatools$ conda create --name datatools python=3.8 -y
datatools$ conda activate datatools
(datatools) datatools$ 
```

## Prerequisites: Building Native Code (C, C++)
Any native code on Linux and macOS builds using `gcc` or `clang` and `make`. On Windows, `Microsoft Visual C++ Build Tools` are used, which can be obtained [here](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16) - or just install Visual Studio.
