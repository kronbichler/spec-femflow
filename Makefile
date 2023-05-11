SRC = $(shell echo femflow.cc ; \
              find dealii/source -name '*.cc' ; \
              find dealii/bundled/boost-1.70.0/libs -name '*.cpp')
OBJ = $(subst .cpp,.o,$(subst .cc,.o,$(SRC)))

INCLUDEDIRS = -Idealii/include \
              -Idealii/bundled/boost-1.70.0/include

%.o : %.cc
	c++ -std=c++17 -march=native -O3 -fopenmp-simd -c $< -o $@ $(INCLUDEDIRS)

%.o : %.cpp
	c++ -std=c++17 -march=native -O3 -c $< -o $@ $(INCLUDEDIRS)

femflow: $(OBJ)
	c++ -o $@ $(OBJ) -lpthread

clean:
	-rm -f $(OBJ) femflow
