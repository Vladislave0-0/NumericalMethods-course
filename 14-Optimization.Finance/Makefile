CXX = g++
CXXFLAGS = -O3 -mavx2 -mfma -fopenmp -I.

LDFLAGS = -lsleef -lm

TARGET = portfolio
SRCS = portfolio.cpp
OBJS_LIB = lib/logf_avx2.o lib/logf_scalar.o

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) $(OBJS_LIB) $(LDFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET)

run: all
	./$(TARGET)

.PHONY: all clean run
