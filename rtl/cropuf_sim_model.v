`timescale 1ns/1ps
// ============================================================
// Module      : cropuf_sim_model
// Project     : NLFSR-Shield CRO-PUF
// File        : rtl/baseline/cropuf_sim_model.v
//
// Description : SIMULATION-ONLY behavioural model of the
//               CRO-PUF, matching model/puf_core.py
//               bit for bit.
//
// WHY THIS FILE EXISTS:
//   Ring oscillators cannot be simulated in behavioural (zero-
//   delay) simulation, because gate propagation delay - the
//   physical mechanism that creates oscillation - does not
//   exist in that simulation mode. Physical oscillation appears
//   only in gate-level simulation with SDF back-annotation.
//   This is a documented limitation and is declared as such in
//   the methodology section of the manuscript.
//
// ------------------------------------------------------------
// CORRECTION IN VERSION 2.0 (first revision)
// ------------------------------------------------------------
//   The previous model gave each ring oscillator a single
//   full-range 16-bit fingerprint and applied a fixed offset of
//   256 counts when the challenge configuration bit was set.
//   Exhaustive testing showed that this offset NEVER altered
//   the comparison outcome on any of the 65,535 challenges,
//   because the gap between two full-range 16-bit fingerprints
//   is typically thousands of counts. The configuration bits
//   were therefore dead logic, and the response depended on
//   only the four pair-selection bits of the challenge; the
//   remaining twelve bits had no influence whatsoever.
//
//   Following the Configurable Inversion Unit of Kareem and
//   Dunaev, which combines an XOR gate, an XNOR gate and a
//   two-to-one multiplexer with three such units per ring
//   oscillator, each oscillator now has THREE configurable
//   stages. Each stage is driven by a challenge bit and
//   contributes a device-specific delay under each of its two
//   configurations. Every challenge bit now influences the
//   response (single-bit sensitivity test in puf_core.py).
// ------------------------------------------------------------
//
// ------------------------------------------------------------
// CHANGE IN VERSION 3.0 (second revision): independent devices
// ------------------------------------------------------------
//   Versions 1 and 2 computed the device parameters inside this
//   module with a 16-bit LCG, and the seed of each further
//   device was the next LCG value after the previous seed. The
//   virtual devices were therefore chained to each other, and
//   the LCG (4259, 47) mod 2^16 is not a full-period generator
//   (4258 is not divisible by 4, so Hull-Dobell fails). See
//   model/legacy_lcg.py.
//
//   From version 3.0 this module contains NO generator. The
//   device parameters are drawn in Python from an independent
//   PCG64 stream per device (puf_core.build_device) and loaded
//   here with $readmemh from two files written by
//   tools/export_device_hex.py:
//
//     BASE_FILE  : 16 words,  base[ro]
//     DELTA_FILE : 96 words,  delta[ro][stage][cfg] at index
//                             ro*6 + stage*2 + cfg
//
//   Each word is a 16-bit two's-complement integer in hex.
//
// MODEL DEFINITION (identical to puf_core.py):
//
//   Stage `s` of oscillator `ro` is driven by challenge bit
//   (ro + 5*s) mod 16. The stride of 5 is coprime with 16, so
//   the three stages of one oscillator read three different
//   challenge bits and, across the sixteen oscillators, every
//   challenge bit drives some stage.
//
//   frequency(ro, ch) = base[ro]
//                     + delta[ro][0][ch[(ro+0)  mod 16]]
//                     + delta[ro][1][ch[(ro+5)  mod 16]]
//                     + delta[ro][2][ch[(ro+10) mod 16]]
//
//   response = 1 if frequency(sel_a) > frequency(sel_b) else 0
//
// INTERFACE: unchanged - drop-in replacement for cropuf_top.
//
// Version : 3.0  (device parameters loaded from file)
// ============================================================

module cropuf_sim_model #(
    parameter NUM_ROS = 16,

    // Device parameter files written by tools/export_device_hex.py.
    // "device_000_*" is device 0, the reference device of the paper.
    parameter BASE_FILE  = "device_000_base.hex",
    parameter DELTA_FILE = "device_000_delta.hex"
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        auto_mode,
    input  wire [15:0] ext_challenge,
    input  wire        ext_start,
    output reg  [15:0] challenge_out,
    output reg         response_out,
    output reg         valid,
    output reg         busy
);

    localparam integer STAGES       = 3;

    // --------------------------------------------------------
    // PROCESS VARIATION MODEL
    // --------------------------------------------------------
    // ro_base[ro]                    nominal frequency offset
    // ro_delta[ro*6 + stage*2 + cfg] delay of one stage under
    //                                one configuration
    reg signed [15:0] ro_base  [0:NUM_ROS-1];
    reg signed [15:0] ro_delta [0:(NUM_ROS*STAGES*2)-1];

    initial begin : device_init
        $readmemh(BASE_FILE,  ro_base);
        $readmemh(DELTA_FILE, ro_delta);
    end

    // --------------------------------------------------------
    // LFSR CHALLENGE GENERATOR
    // --------------------------------------------------------
    // x^16 + x^14 + x^13 + x^11 + 1, seed 0xACE1.
    // Identical to crp_controller.v and to puf_core.py.
    reg  [15:0] lfsr_reg;
    wire        lfsr_fb;
    assign lfsr_fb = lfsr_reg[15] ^ lfsr_reg[13]
                   ^ lfsr_reg[12] ^ lfsr_reg[10];

    reg [15:0] current_challenge;

    localparam S_IDLE    = 2'b00;
    localparam S_COMPUTE = 2'b01;
    localparam S_OUTPUT  = 2'b10;
    reg [1:0]  state;

    // --------------------------------------------------------
    // PAIR SELECTION - mirrors cro_puf_core.v
    // --------------------------------------------------------
    wire [3:0] sel_a;
    wire [3:0] sel_b;
    assign sel_a = current_challenge[3:0];
    assign sel_b = current_challenge[3:0] ^ 4'b0001;

    // --------------------------------------------------------
    // EFFECTIVE FREQUENCY OF ONE OSCILLATOR
    // --------------------------------------------------------
    function signed [17:0] ro_frequency;
        input [3:0]  ro;
        input [15:0] ch;
        reg   [3:0]  p0, p1, p2;
        reg          c0, c1, c2;
        begin
            p0 = ro + 4'd0;         // 4-bit add wraps mod 16
            p1 = ro + 4'd5;
            p2 = ro + 4'd10;
            c0 = ch[p0];
            c1 = ch[p1];
            c2 = ch[p2];
            ro_frequency = ro_base[ro]
                         + ro_delta[(ro*6) + 0 + c0]
                         + ro_delta[(ro*6) + 2 + c1]
                         + ro_delta[(ro*6) + 4 + c2];
        end
    endfunction

    wire signed [17:0] freq_a;
    wire signed [17:0] freq_b;
    assign freq_a = ro_frequency(sel_a, current_challenge);
    assign freq_b = ro_frequency(sel_b, current_challenge);

    wire computed_response;
    assign computed_response = (freq_a > freq_b) ? 1'b1 : 1'b0;

    // --------------------------------------------------------
    // MAIN FSM - 3 clock cycles per CRP
    // --------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state             <= S_IDLE;
            lfsr_reg          <= 16'hACE1;
            challenge_out     <= 16'h0000;
            current_challenge <= 16'h0000;
            response_out      <= 1'b0;
            valid             <= 1'b0;
            busy              <= 1'b0;
        end else begin
            valid <= 1'b0;

            case (state)

                S_IDLE: begin
                    busy <= 1'b0;
                    if (auto_mode || ext_start) begin
                        if (auto_mode) begin
                            current_challenge <= lfsr_reg;
                            challenge_out     <= lfsr_reg;
                            lfsr_reg <= {lfsr_reg[14:0], lfsr_fb};
                        end else begin
                            current_challenge <= ext_challenge;
                            challenge_out     <= ext_challenge;
                        end
                        busy  <= 1'b1;
                        state <= S_COMPUTE;
                    end
                end

                S_COMPUTE: begin
                    response_out <= computed_response;
                    state        <= S_OUTPUT;
                end

                S_OUTPUT: begin
                    valid <= 1'b1;
                    busy  <= 1'b0;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;

            endcase
        end
    end

endmodule
