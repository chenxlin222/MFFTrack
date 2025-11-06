from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        name='videoanalyst.engine.tester.tester_impl.utils',
        sources=[
            'videoanalyst/engine/tester/tester_impl/utils/region.pyx',
            'videoanalyst/engine/tester/tester_impl/utils/src/region.c',
        ],
        include_dirs=[
            'videoanalyst/engine/tester/tester_impl/utils/src'
        ]
    )
]

setup(
    name='videoanalyst',
    packages=['videoanalyst'],
    ext_modules=cythonize(ext_modules)
)