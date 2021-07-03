.PHONY: all clean

SRC_FILES=$(wildcard *.cpp)
HEADER_FILES=$(wildcard *.h)

all: $(SRC_FILES) $(HEADER_FILES)
	g++ -std=c++17 -march=native -O3 -fopenmp -fstrict-aliasing $(SRC_FILES) -o copytest

clean:
	rm -rf *.o
	rm -rf copytest
	
