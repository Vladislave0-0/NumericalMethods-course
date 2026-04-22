#include "graphics.h"

void print_table(test_result *results, int count) {
  printf("\n" BLU "┌────────────────┬──────────────┬────────┬────────"
         "┬──────────┐" RST "\n");
  printf(BLU "│" RST " %-14s " BLU "│" RST " %-12s " BLU "│" RST " %-6s " BLU
             "│" RST " %-6s " BLU "│" RST " %-8s " BLU "│" RST "\n",
         "   Input x", "Output ln(x)", " Errno", " Flag", " Status");
  printf(BLU "├────────────────┼──────────────┼────────┼────────┼────"
             "──────┤" RST "\n");

  for (int i = 0; i < count; ++i) {
    printf(BLU "│" RST " %-14s " BLU "│" RST " %-12s " BLU "│" RST " %-6s " BLU
               "│" RST " %-6s " BLU "│" RST " %s%-9s" RST BLU "│" RST "\n",
           results[i].label, results[i].output,
           results[i].errno_ok ? "  OK" : RED "  ERR" RST,
           results[i].flag_ok ? "  OK" : RED "  ERR" RST,
           results[i].status_ok ? GRN : RED,
           results[i].status_ok ? "  PASS" : "  FAIL");
  }

  printf(BLU "└────────────────┴──────────────┴────────┴────────┴────"
             "──────┘" RST "\n");
}
