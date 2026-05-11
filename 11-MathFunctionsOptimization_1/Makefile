CXX = g++
CXXFLAGS = -O3 -Iinclude -march=native -std=c++20
LDFLAGS = -lm

SRC_DIR = src
OBJ_DIR = obj
LIB_DIR = lib
TARGET = test

CUSTOM_OBJS = $(LIB_DIR)/logf_scalar.o $(LIB_DIR)/logf_avx2.o

# MKL
ifeq ($(MKL), 1)
    CXXFLAGS += -DUSE_MKL -I$(MKLROOT)/include
    LDFLAGS += -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -ldl
endif

# SLEEF
ifeq ($(SLEEF), 1)
    CXXFLAGS += -DUSE_SLEEF
    LDFLAGS += -lsleef
endif

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