#pragma once

#include <bits/stdint-intn.h>
#include <bits/stdint-uintn.h>

#define COARSENING_FACTOR 5
#define BLOCK_DIM 256
#define BLOCK_DIM_STR 128

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n)((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

// unsigned int BLOCK_DIM  = 256;
// unsigned int COARSENING_FACTOR = 5;