#ifndef GRAPHICS_H
#define GRAPHICS_H

#include <stdio.h>

typedef struct {
  char label[24];
  char output[16];
  int errno_ok;
  int flag_ok;
  int status_ok;
} test_result;

#define RED "\x1B[31m"
#define GRN "\x1B[32m"
#define BLU "\x1B[34m"
#define RST "\x1B[0m"

void print_table(test_result *results, int count);

#endif // GRAPHICS_H
