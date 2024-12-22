from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension(
        "ringtrap_cython.potentials",
        sources =[
            "./ringtrap_cython/potentials.pyx",
            "./ringtrap_cython/potentials_lib.cpp"
        ],
        language="c++",
        include_dirs=["."],
        extra_compile_args=["-w"]
    )
]

setup(
    ext_modules=cythonize(extensions, language_level="3")
)
