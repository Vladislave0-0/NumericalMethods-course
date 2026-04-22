TARGET = ./bin/test_runner

CC = gcc
CFLAGS = -Wall -Wextra -I./include -O2
LDFLAGS = -lm

SRC_DIR = ./src
OBJ_DIR = ./obj
BIN_DIR = ./bin
LOGF_OBJ = ./lib/logf.o

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

all: $(BIN_DIR) $(OBJ_DIR) $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) $(LOGF_OBJ) -o $(TARGET) $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean
