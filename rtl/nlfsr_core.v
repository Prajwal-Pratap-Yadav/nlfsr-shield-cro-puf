`timescale 1ns/1ps
// ============================================================
// Module      : nlfsr_core
// Project     : NLFSR-Shield CRO-PUF
// File        : rtl/nlfsr/nlfsr_core.v
//
// Description : 32-bit Non-Linear Feedback Shift Register.
//               Core primitive of the NLFSR-Shield layer.
//
// ------------------------------------------------------------
// CORRECTION IN THIS VERSION
// ------------------------------------------------------------
//   The maximally non-linear combining function was previously
//   applied to taps {31, 15, 7}. Bit 31 is also the bit that
//   is shifted out of the register, so the feedback expanded to
//
//     fb = s31*(1 ^ s15 ^ s7) ^ (s21 ^ s1 ^ s0 ^ s15*s7)
//
//   The coefficient of s31 vanishes whenever s15 ^ s7 = 1, i.e.
//   over exactly half the state space. For those states the
//   successor does not depend on s31, two distinct predecessors
//   share a successor, and the state map is NOT a bijection.
//   Exhaustive cycle detection showed that every device seed
//   converged to a single cycle of length 1,832, and that
//   0x00000000 and 0xFFFFFFFF were absorbing fixed points.
//
//   The first tap has been moved from bit 31 to bit 30 so that
//   the outgoing bit appears only in the linear term. The state
//   map is then a bijection, verified directly. Exhaustive
//   traversal of the cycle containing each device's seeded
//   state gives periods between 6.12e8 and 2.88e9.
//
//   Do not move this tap back to bit 31, and do not place any
//   product term on the outgoing bit of the register.
// ------------------------------------------------------------
//
// WHY NLFSR OVER LFSR:
//   A standard LFSR uses linear feedback, which is easily
//   modelled by logistic regression and recovered outright by
//   the Berlekamp-Massey algorithm from 2*WIDTH output bits.
//   The non-linear term removes that direct recovery.
//
// DESIGN PARAMETERS:
//   Register width : 32 bits
//   Period         : determined by exhaustive traversal, not
//                    assumed. 6.12e8 to 2.88e9 depending on the
//                    seeded state. The shortest observed period
//                    exceeds the largest dataset used in this
//                    work by a factor of 612.
//
//   Feedback polynomial (linear taps):
//     x^32 + x^22 + x^2 + x^1 + 1, taps {31, 21, 1, 0}.
//     A primitive polynomial over GF(2). Note that primitivity
//     guarantees a maximal period only for the purely linear
//     register; once a non-linear term is added the period must
//     be established empirically, which is what we do.
//
//   Maximally non-linear (MNL) combining function:
//     f(a,b,c) = ab XOR bc XOR ac, applied to taps {30, 15, 7}.
//     This function attains non-linearity 2, the maximum
//     available to any three-variable Boolean function. It is
//     NOT a bent function: bent functions are defined only for
//     an even number of variables, and this one has three
//     inputs. The manuscript uses the term "maximally
//     non-linear" for this reason.
//
//   Seeding: the first 32 response bits from the CRO-PUF seed
//            the state register. No external seed required.
//
// Version : 2.1  (v2.0: MNL tap corrected from bit 31 to bit 30;
//                v2.1: advance input, one step per PUF response)
// ============================================================

module nlfsr_core #(
    parameter WIDTH = 32
)(
    input  wire             clk,
    input  wire             rst_n,

    // seed_mode=1: loading seed bits from PUF output
    // seed_mode=0: running freely, generating keystream
    input  wire             seed_mode,
    input  wire             seed_bit,    // one seed bit per cycle

    // advance=1: step the running register by one position.
    // Added in v2.1 so that the register advances exactly once per
    // PUF response (as in puf_core.NLFSRModel), not on every clock.
    // Tie to 1'b1 for a free-running keystream.
    input  wire             advance,

    // Output: one obfuscation bit per clock cycle
    output wire             nlfsr_out,
    output reg              ready        // high when seeded
);

    // --------------------------------------------------------
    // STATE REGISTER
    // --------------------------------------------------------
    // state[WIDTH-1] is the outgoing end; state[0] is where the
    // feedback bit is injected on each shift.
    reg [WIDTH-1:0] state;

    // --------------------------------------------------------
    // SEED COUNTER
    // --------------------------------------------------------
    reg [5:0] seed_count;    // counts 0 to 32

    // --------------------------------------------------------
    // LINEAR FEEDBACK TAPS
    // --------------------------------------------------------
    // Polynomial x^32 + x^22 + x^2 + x^1 + 1
    //   bit 31 (x^32), bit 21 (x^22), bit 1 (x^2), bit 0 (x^1)
    //
    // Bit 31 appears here, and ONLY here. Its coefficient in the
    // combined feedback is therefore the constant 1, which is
    // what makes the state map invertible.
    wire linear_fb;
    assign linear_fb = state[31] ^ state[21]
                     ^ state[1]  ^ state[0];

    // --------------------------------------------------------
    // MAXIMALLY NON-LINEAR COMBINING FUNCTION
    // --------------------------------------------------------
    // f(a, b, c) = ab XOR bc XOR ac
    //
    // Non-linearity 2, the maximum attainable by a three-
    // variable Boolean function, so the function is at the
    // greatest possible Hamming distance from every affine
    // function over GF(2)^3.
    //
    // Tap positions:
    //   a = state[30]  - CORRECTED from state[31]; see header
    //   b = state[15]  - middle tap
    //   c = state[7]   - quarter tap
    wire a, b, c;
    assign a = state[30];      // was state[31]; see header note
    assign b = state[15];
    assign c = state[7];

    wire mnl_fb;
    assign mnl_fb = (a & b) ^ (b & c) ^ (a & c);

    // Combined feedback: linear XOR non-linear.
    wire combined_fb;
    assign combined_fb = linear_fb ^ mnl_fb;

    // --------------------------------------------------------
    // OUTPUT TAP
    // --------------------------------------------------------
    // The combined feedback bit is emitted directly rather than
    // tapping state[WIDTH-1]. This bit is recomputed every clock
    // cycle from the whole state, so it cannot remain constant
    // across consecutive cycles in the way that a single state
    // bit can when the register holds a run-length pattern.
    assign nlfsr_out = combined_fb;

    // --------------------------------------------------------
    // MAIN SEQUENTIAL LOGIC
    // --------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state      <= {WIDTH{1'b0}};
            seed_count <= 6'd0;
            ready      <= 1'b0;

        end else if (seed_mode && !ready) begin
            // SEED LOADING: shift one PUF response bit into the
            // LSB per cycle. After WIDTH cycles the register
            // holds a device-unique seed derived from the
            // physical process variation fingerprint.
            state      <= {state[WIDTH-2:0], seed_bit};
            seed_count <= seed_count + 1;

            if (seed_count == WIDTH - 1) begin
                ready <= 1'b1;
            end

        end else if (ready && advance) begin
            // RUNNING: shift left, inject the combined feedback.
            state <= {state[WIDTH-2:0], combined_fb};
        end
    end

endmodule
