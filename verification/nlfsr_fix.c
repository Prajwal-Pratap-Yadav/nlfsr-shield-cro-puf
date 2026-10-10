/* Search for a corrected NLFSR feedback whose state map is a bijection
 * and whose period is long, then verify the period exhaustively. */
#include <stdio.h>
#include <stdint.h>

typedef struct { int a, b, c; const char *name; } Variant;

static inline uint32_t lin_part(uint32_t s) {
    return ((s >> 31) ^ (s >> 21) ^ (s >> 1) ^ s) & 1u;   /* x^32+x^22+x^2+x+1 */
}

static inline uint32_t stepv(uint32_t s, int A, int B, int C) {
    uint32_t lin = lin_part(s);
    if (A < 0) return (s << 1) | lin;                     /* pure LFSR */
    uint32_t a = (s >> A) & 1u, b = (s >> B) & 1u, c = (s >> C) & 1u;
    uint32_t mnl = (a & b) ^ (b & c) ^ (a & c);
    return (s << 1) | (lin ^ mnl);
}

/* bijection test: does the successor ever fail to depend on bit 31? */
static int is_bijection(int A, int B, int C) {
    if (A < 0) return 1;                    /* pure LFSR is always a bijection */
    if (A == 31 || B == 31 || C == 31) return 0;  /* s31 enters a product term */
    return 1;
}

/* walk the cycle containing seed; cap the number of steps */
static uint64_t cycle_len(uint32_t seed, int A, int B, int C, uint64_t cap) {
    uint32_t s = seed;
    uint64_t n = 0;
    do {
        s = stepv(s, A, B, C);
        n++;
        if (n > cap) return 0;              /* exceeded cap */
    } while (s != seed);
    return n;
}

int main(void) {
    Variant vs[] = {
        { -1, -1, -1, "pure 32-bit LFSR (no MNL term)          " },
        { 31, 15,  7, "as published: MNL taps 31, 15, 7        " },
        { 30, 15,  7, "MNL taps 30, 15, 7                      " },
        { 25, 15,  7, "MNL taps 25, 15, 7                      " },
        { 29, 19,  9, "MNL taps 29, 19, 9                      " },
        { 24, 13,  3, "MNL taps 24, 13, 3                      " },
    };
    const uint64_t CAP = 5000000000ULL;
    const uint64_t MAXP = 4294967295ULL;

    printf("%-42s %-10s %14s %10s\n", "variant", "bijection", "period(seed)", "= 2^32-1?");
    for (unsigned i = 0; i < sizeof(vs)/sizeof(vs[0]); i++) {
        int bij = is_bijection(vs[i].a, vs[i].b, vs[i].c);
        uint64_t p = cycle_len(0x0000BEEFu, vs[i].a, vs[i].b, vs[i].c, CAP);
        printf("%-42s %-10s %14llu %10s\n", vs[i].name,
               bij ? "yes" : "NO",
               (unsigned long long)p,
               (p == MAXP) ? "YES" : "no");
        fflush(stdout);
    }

    printf("\nNote: for a non-bijective map the seed may sit on a tail rather than\n");
    printf("a cycle, in which case the walk never returns and 0 is reported.\n");
    return 0;
}
