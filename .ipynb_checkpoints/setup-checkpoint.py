from setuptools import setup, find_packages

setup(
    name="PILOT-GM-VAE",
    version="0.1",
    packages=find_packages(),
    author="Mehdi Joodaki, Mina Shaigan",
    author_email="judakimehdi@gmail.com, mina.shaigan@gmail.com",
    description="Patient-Level Analysis of Single Cell Disease Atlas with Optimal Transport of Gaussian Mixtures (PILOT_VAE)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/minashaigan/PILOT_VAE",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11.5,<3.12',
    install_requires=[
        "adjusttext==0.8",
        "annotated-types==0.7.0",
        "array-api-compat==1.8",
        "build==1.2.1",
        "cachecontrol==0.14.0",
        "cachetools==5.3.3",
        "cftime==1.6.2",
        "chardet==5.2.0",
        "chex==0.1.87",
        "cleo==2.1.0",
        "cloudpickle==2.2.1",
        "cmake==3.28.3",
        "colorama==0.4.6",
        "colorcet==3.1.0",
        "colour==0.1.5",
        "contourpy==1.1.1",
        "crashtest==0.4.1",
        "dagstream==0.1.5",
        "deprecated==1.2.14",
        "distlib==0.3.8",
        "docutils==0.21.2",
        "dulwich==0.21.7",
        "elpigraph-python==0.3.1",
        "etils==1.9.4",
        "fastjsonschema==2.19.1",
        "filelock==3.14.0",
        "fonttools==4.42.1",
        "fsspec==2024.9.0",
        "future==0.18.3",
        "gprofiler-official==1.0.0",
        "graphtools==1.5.3",
        "h11==0.14.0",
        "h5py==3.9.0",
        "humanize==4.10.0",
        "igraph==0.10.8",
        "importlib-metadata==7.1.0",
        "importlib-resources==6.4.5",
        "iniconfig==2.0.0",
        "installer==0.7.0",
        "itsdangerous==2.2.0",
        "jaraco-classes==3.4.0",
        "jax==0.4.34",
        "jaxlib==0.4.34",
        "jeepney==0.8.0",
        "joblib==1.3.2",
        "joypy==0.2.6",
        "keyring==24.3.1",
        "kiwisolver==1.4.5",
        "leidenalg==0.10.1",
        "llvmlite==0.40.1",
        "logomaker==0.8",
        "magic-impute==3.0.0",
        "marimo==0.7.12",
        "matplotlib==3.8.0",
        "mizani==0.9.3",
        "ml-dtypes==0.5.0",
        "more-itertools==10.2.0",
        "multipledispatch==1.0.0",
        "natsort==8.4.0",
        "netcdf4==1.6.4",
        "networkx==3.1",
        "nh3==0.2.17",
        "numba==0.57.1",
        "numexpr==2.8.6",
        "numpy==1.24.4",
        "nvidia-cublas-cu12==12.1.3.1",
        "nvidia-cuda-cupti-cu12==12.1.105",
        "nvidia-cuda-nvrtc-cu12==12.1.105",
        "nvidia-cuda-runtime-cu12==12.1.105",
        "nvidia-cufft-cu12==11.0.2.54",
        "nvidia-curand-cu12==10.3.2.106",
        "nvidia-nccl-cu12==2.20.5",
        "nvidia-nvjitlink-cu12==12.6.77",
        "nvidia-nvtx-cu12==12.1.105",
        "opencv-python==4.9.0.80",
        "packaging==24.0",
        "pandas==2.2.2",
        "patsy==0.5.3",
        "pillow==10.0.1",
        "pilotpy==2.0.6",
        "pipe==2.2",
        "pkginfo==1.10.0",
        "platformdirs==4.2.2",
        "plotly==5.22.0",
        "plotnine==0.12.3",
        "pluggy==1.5.0",
        "poetry==1.8.2",
        "poetry-core==1.9.0",
        "poetry-plugin-export==1.7.1",
        "pooch==1.8.2",
        "pot==0.9.1",
        "pycryptodomex==3.20.0",
        "pydantic==2.9.2",
        "pydantic-core==2.23.4",
        "pydiffmap==0.2.0.1",
        "pydot==3.0.1",
        "pygraphviz==1.13",
        "pygsp==0.5.1",
        "pymdown-extensions==10.9",
        "pynndescent==0.5.10",
        "pyparsing==3.1.1",
        "pyproject-api==1.6.1",
        "pyproject-hooks==1.1.0",
        "pytest==8.2.1",
        "python-git==2018.2.1",
        "python-igraph==0.10.8",
        "python-louvain==0.16",
        "pytorch-ignite==0.5.1",
        "pyvista==0.43.10",
        "pyyaml==6.0.2",
        "pyzmq==25.1.2",
        "rapidfuzz==3.9.0",
        "readme-renderer==43.0",
        "requests-toolbelt==1.0.0",
        "researchpy==0.3.5",
        "rfc3986==2.0.0",
        "rich==13.7.1",
        "rpds-py==0.18.0",
        "ruff==0.5.5",
        "s-gd2==1.8.1",
        "scanpy==1.9.5",
        "scikit-learn==1.5.2",
        "scikit-network==0.31.0",
        "scikit-sparse==0.4.15",
        "scipy==1.11.2",
        "scooby==0.10.0",
        "scprep==1.1.0",
        "scvi-colab==0.12.0",
        "seaborn==0.12.2",
        "secretstorage==3.3.3",
        "send2trash==1.8.2",
        "session-info==1.0.0",
        "shap==0.42.1",
        "shapely==2.0.1",
        "shellingham==1.5.4",
        "shutup==0.2.0",
        "slicer==0.0.7",
        "snowballstemmer==2.2.0",
        "sparse==0.15.4",
        "sphinxcontrib-applehelp==1.0.8",
        "sphinxcontrib-devhelp==1.0.6",
        "sphinxcontrib-htmlhelp==2.0.5",
        "sphinxcontrib-jsmath==1.0.1",
        "sphinxcontrib-qthelp==1.0.7",
        "sphinxcontrib-serializinghtml==1.1.10",
        "starlette==0.38.2",
        "statsmodels==0.14.0",
        "stdlib-list==0.9.0",
        "sympy==1.13.3",
        "tabulate==0.9.0",
        "tasklogger==1.2.0",
        "tbb==2021.10.0",
        "tenacity==8.2.3",
        "tensorstore==0.1.66",
        "texttable==1.6.7",
        "threadpoolctl==3.2.0",
        "tomlkit==0.12.4",
        "torch==2.4.1",
        "tox==4.15.0",
        "triton==3.0.0",
        "trove-classifiers==2024.4.10",
        "twine==5.0.0",
        "types-python-dateutil==2.8.19.20240106",
        "typing-extensions==4.12.2",
        "umap-learn==0.5.4",
        "uri-template==1.3.0",
        "uvicorn==0.30.3",
        "virtualenv==20.26.1",
        "vtk==9.3.1",
        "webcolors==1.13",
        "websockets==12.0",
        "zipp==3.18.1",
        ]
)
