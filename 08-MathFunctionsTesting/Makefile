CC = gcc
CFLAGS = -Wall -Wextra -I./include -O3 -mavx2 -mfma
LDFLAGS = -lm

SRC_DIR = ./src
OBJ_DIR = ./obj
BIN_DIR = ./bin
LIB_DIR = ./lib

TARGET_SCALAR = $(BIN_DIR)/test_scalar
TARGET_VECTOR = $(BIN_DIR)/test_avx2

LIB_SCALAR = $(LIB_DIR)/logf_scalar.o
LIB_VECTOR = $(LIB_DIR)/logf_avx2.o $(LIB_DIR)/logf_scalar.o

COMMON_OBJS = $(OBJ_DIR)/graphics.o

all: $(BIN_DIR) $(OBJ_DIR) $(TARGET_SCALAR) $(TARGET_VECTOR)

$(TARGET_SCALAR): $(OBJ_DIR)/test_scalar.o $(COMMON_OBJS)
	$(CC) $(CFLAGS) $^ $(LIB_SCALAR) -o $@ $(LDFLAGS)

$(TARGET_VECTOR): $(OBJ_DIR)/test_avx2.o $(COMMON_OBJS)
	$(CC) $(CFLAGS) $^ $(LIB_VECTOR) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean
