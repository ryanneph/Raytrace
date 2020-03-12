.PHONY: all build test clean

all: build

build:
		@echo "Building Cython Extension in place"
		python setup.py build_ext --inplace

test:
		@echo "Testing raytrace extension"
		python test/test.py

clean:
	  rm -f raytrace.cpython-* raytrace.cpp
