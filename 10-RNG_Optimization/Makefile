CXX      := g++
CXXFLAGS := -O3 -march=native -fopenmp -ffast-math -DNDEBUG -I./include
LDFLAGS  := -fopenmp

SRC_DIR  := src
OBJ_DIR  := obj
INC_DIR  := include

TARGETS  := verify_bitwise monte_carlo_vec

OBJ_VERIFY := $(OBJ_DIR)/verify_bitwise.o
OBJ_MC     := $(OBJ_DIR)/monte_carlo_vec.o

all: $(TARGETS)

verify_bitwise: $(OBJ_VERIFY)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

monte_carlo_vec: $(OBJ_MC)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(TARGETS)

.PHONY: all clean
