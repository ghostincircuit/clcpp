all:
	g++ base.cc -std=c++11 -O0 -g -rdynamic -lOpenCL -o base
clean:
	rm -f base
