from pathlib import Path
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import numpy as np

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

ext_modules = [
    Pybind11Extension(
        "two_layer_hnsw_like_cpp._tlhl_cpp",
        ["src/tlhl_core.cpp"],
        include_dirs=[np.get_include()],
        cxx_std=17,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    name="two-layer-hnsw-like-cpp",
    version="0.1.0",
    description="Two-layer HNSW-like ANN index with C++ core and pybind11",
    long_description=README,
    long_description_content_type="text/markdown",
    author="OpenAI",
    packages=["two_layer_hnsw_like_cpp"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scikit-learn>=1.2",
        "faiss-cpu>=1.7.4",
    ],
    extras_require={
        "dev": ["pytest>=8.0"],
    },
    include_package_data=False,
    zip_safe=False,
)
