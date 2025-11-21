pip intsall -r requirements-dev.txt
cmake --preset linux-gcc
cmake --build --preset linux-gcc-release
pip install .
