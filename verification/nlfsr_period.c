/* Exhaustive verification of the NLFSR-Shield 32-bit state machine.
 *
 * Feedback (Section 3.3.2 of the manuscript):
 *   linear taps at 31, 21, 1, 0   (x^32 + x^22 + x^2 + x + 1)
 *   MNL f(a,b,c) = ab ^ bc ^ ac   at positions 31, 15, 7
 *   fb = linear ^ mnl
 *   s' = (s << 1) | fb
 *
 * Determines (a) whether the state map is a bijection, and
 * (b) the tail length and cycle length reachable from given seeds,
 * using Brent's cycle-detection algorithm.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline uint32_t step(uint32_t s) {
    uint32_t lin = ((s >> 31) ^ (s >> 21) ^ (s >> 1) ^ s) & 1u;
    uint32_t a = (s >> 31) & 1u, b = (s >> 15) & 1u, c = (s >> 7) & 1u;
    uint32_t mnl = (a & b) ^ (b & c) ^ (a & c);
    return (s << 1) | (lin ^ mnl);
}

/* Brent's algorithm: returns tail length mu and cycle length lambda. */
static void brent(uint32_t x0, uint64_t *mu_out, uint64_t *lam_out) {
    uint64_t power = 1, lam = 1;
    uint32_t tortoise = x0, hare = step(x0);
    while (tortoise != hare) {
        if (power == lam) { tortoise = hare; power <<= 1; lam = 0; }
        hare = step(hare);
        lam++;
    }
    uint64_t mu = 0;
    tortoise = x0; hare = x0;
    for (uint64_t i = 0; i < lam; i++) hare = step(hare);
    while (tortoise != hare) { tortoise = step(tortoise); hare = step(hare); mu++; }
    *mu_out = mu; *lam_out = lam;
}

int main(void) {
    /* ---- (1) is the state map a bijection? ---- */
    /* s' determines s30..s0; s31 is recoverable only if fb depends on s31.
       coefficient of s31 in fb is (1 ^ s15 ^ s7).                        */
    printf("=== Injectivity analysis ===\n");
    printf("fb = s31*(1 ^ s15 ^ s7) ^ (s21 ^ s1 ^ s0 ^ s15*s7)\n");
    printf("The coefficient of s31 vanishes whenever s15 ^ s7 = 1.\n");
    printf("For those states the successor does not depend on s31, so two\n");
    printf("distinct predecessors map to the same successor.\n\n");

    /* empirical confirmation on a sample: count collisions */
    uint64_t collisions = 0, tested = 0;
    for (uint64_t i = 0; i < 40000000ULL; i++) {
        uint32_t s = (uint32_t)(i * 2654435761u + 12345u);
        uint32_t t = s ^ 0x80000000u;          /* flip s31 only */
        if (step(s) == step(t)) collisions++;
        tested++;
    }
    printf("sampled %llu states: %llu had step(s) == step(s ^ 2^31)  (%.4f%%)\n",
           (unsigned long long)tested, (unsigned long long)collisions,
           100.0 * (double)collisions / (double)tested);
    printf("=> the map is NOT a bijection; a maximal 2^32-1 cycle is impossible.\n\n");

    /* ---- (2) cycle structure from representative seeds ---- */
    printf("=== Brent cycle detection from representative seeds ===\n");
    printf("%-14s %14s %14s\n", "seed", "tail (mu)", "cycle (lambda)");

    uint32_t seeds[] = {
        0x00000001u, 0x0000BEEFu, 0xACE1ACE1u, 0xFFFFFFFFu,
        0x12345678u, 0xDEADBEEFu, 0xA5A5A5A5u, 0x00000000u
    };
    uint64_t maxlam = 0;
    for (unsigned i = 0; i < sizeof(seeds)/sizeof(seeds[0]); i++) {
        uint64_t mu, lam;
        brent(seeds[i], &mu, &lam);
        printf("0x%08X   %14llu %14llu\n", seeds[i],
               (unsigned long long)mu, (unsigned long long)lam);
        fflush(stdout);
        if (lam > maxlam) maxlam = lam;
    }

    /* random seeds */
    srand(12345);
    for (int i = 0; i < 12; i++) {
        uint32_t s = ((uint32_t)rand() << 17) ^ ((uint32_t)rand() << 3) ^ (uint32_t)rand();
        uint64_t mu, lam;
        brent(s, &mu, &lam);
        printf("0x%08X   %14llu %14llu\n", s,
               (unsigned long long)mu, (unsigned long long)lam);
        fflush(stdout);
        if (lam > maxlam) maxlam = lam;
    }

    printf("\nlongest cycle observed: %llu\n", (unsigned long long)maxlam);
    printf("2^32 - 1 = 4294967295\n");
    printf("ratio observed/claimed = %.6f\n", (double)maxlam / 4294967295.0);
    return 0;
}
