/* nlfsr_period_seeds.c - period of the corrected 32-bit NLFSR for each
 * seed listed in a file (one 32-bit hex value per line). The state map is
 * a bijection, so every seed lies on a cycle; its length is the period of
 * the keystream for that device. Used for the per-device periods (R4.4/R4.6).
 * Build: gcc -O3 -o nlfsr_period_seeds nlfsr_period_seeds.c
 * Run:   ./nlfsr_period_seeds ../experiments/device_seeds.txt              */
#include <stdio.h>
#include <stdint.h>
static inline uint32_t step(uint32_t s) {
    uint32_t lin = (s >> 31) ^ (s >> 21) ^ (s >> 1) ^ s;
    uint32_t a = (s >> 30) & 1u, b = (s >> 15) & 1u, c = (s >> 7) & 1u;
    return (s << 1) | ((lin ^ ((a & b) ^ (b & c) ^ (a & c))) & 1u);
}
int main(int argc, char **argv) {
    FILE *f = fopen(argc > 1 ? argv[1] : "device_seeds.txt", "r");
    if (!f) { perror("seed file"); return 1; }
    unsigned int seed; int d = 0;
    while (fscanf(f, "%x", &seed) == 1) {
        uint32_t s = seed; uint64_t n = 0;
        do { s = step(s); n++; } while (s != seed && n < 5000000000ULL);
        printf("device %d seed %08X period %llu\n", d++, seed, (unsigned long long)n);
        fflush(stdout);
    }
    fclose(f);
    return 0;
}
