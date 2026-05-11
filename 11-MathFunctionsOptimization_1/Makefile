CXX = g++
CXXFLAGS = -O3 -Iinclude -march=native -std=c++20
LDFLAGS = -lm

SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
LIB_DIR = lib
TARGET = bench_tester

CUSTOM_OBJS = $(LIB_DIR)/logf_scalar.o $(LIB_DIR)/logf_avx2.o

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))

all: $(TARGET)

$(TARGET): $(OBJS) $(CUSTOM_OBJS)
	$(CXX) $(OBJS) $(CUSTOM_OBJS) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean
