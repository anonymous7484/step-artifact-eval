#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline uint32_t float_to_bits(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    return bits;
}

void init_target(uint32_t* target, int tile_size) {
    for (int i = 0; i < tile_size; ++i) {
        for (int j = 0; j < tile_size; ++j) {
            target[i * tile_size + j] = 0;
        }
    }
}

void print_tile(uint32_t* tile, int tile_size) {
    for (int i = 0; i < tile_size; ++i) {
        for (int j = 0; j < tile_size; ++j) {
            printf("%d ", tile[i * tile_size + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void add_tile_c(uint32_t* resultptr, uint32_t a[], uint32_t b[], int tile_size) {
    init_target(resultptr, tile_size);
    for (int i = 0; i < tile_size; ++i) {
        for (int j = 0; j < tile_size; ++j) {
            float aval = (float)a[i * tile_size + j];
            float bval = (float)b[i * tile_size + j];
            float res = aval + bval;
            resultptr[i * tile_size + j] = float_to_bits(res);
        }
    }
    // print_tile(resultptr, tile_size);
}

void sub_tile_c(uint32_t* resultptr, uint32_t a[], uint32_t b[], int tile_size) {
    init_target(resultptr, tile_size);
    for (int i = 0; i < tile_size; ++i) {
        for (int j = 0; j < tile_size; ++j) {
            float aval = (float)a[i * tile_size + j];
            float bval = (float)b[i * tile_size + j];
            float res = aval - bval;
            resultptr[i * tile_size + j] = float_to_bits(res);
        }
    }
    // print_tile(resultptr, tile_size);
}

void mul_tile_c(uint32_t* resultptr, uint32_t a[], uint32_t b[], int tile_size) {
    init_target(resultptr, tile_size);
    for (int i = 0; i < tile_size; ++i) {
        for (int j = 0; j < tile_size; ++j) {
            float aval = (float)a[i * tile_size + j];
            float bval = (float)b[i * tile_size + j];
            float res = aval * bval;
            resultptr[i * tile_size + j] = float_to_bits(res);
        }
    }
}

void div_tile_c(uint32_t* resultptr, uint32_t a[], uint32_t b[], int tile_size) {
    init_target(resultptr, tile_size);
    for (int i = 0; i < tile_size; ++i) {
        for (int j = 0; j < tile_size; ++j) {
            float aval = (float)a[i * tile_size + j];
            float bval = (float)b[i * tile_size + j];
            float res = aval / bval;
            resultptr[i * tile_size + j] = float_to_bits(res);
        }
    }
    // print_tile(resultptr, tile_size);
}

// Flattened vectors: row-major
void matmul_t_tile_c(uint32_t* resultptr, uint32_t a[], uint32_t b[], int tile_size) {
    init_target(resultptr, tile_size);
    for (int i = 0; i < tile_size; ++i) {
        for (int j = 0; j < tile_size; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < tile_size; ++k) {
                float aval = (float)a[i * tile_size + k];
                float bval = (float)b[j * tile_size + k];
                acc += aval * bval;
            }
            resultptr[i * tile_size + j] = float_to_bits(acc);
        }
    }
    // print_tile(resultptr, tile_size);
    return;
}

void silu_tile_c(uint32_t* resultptr, uint32_t a[], int tile_size) {
    init_target(resultptr, tile_size);
    for (int i = 0; i < tile_size; ++i) {
        for (int j = 0; j < tile_size; ++j) {
            float val = (float)a[i * tile_size + j];
            float res = val / (1.0f + expf(-val));
            resultptr[i * tile_size + j] = float_to_bits(res);
        }
    }
    // printf("silu_tile_c\n");
    // print_tile(resultptr, tile_size);
    return;
}

#ifdef __cplusplus
}
#endif

