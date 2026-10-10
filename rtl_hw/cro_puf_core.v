`timescale 1ns/1ps
// ============================================================
// cro_puf_core.v - complete CRO-PUF with a real ring-oscillator
//                  bank (synthesisable; Section 3.2, Fig. 1)
//
// 16 configurable ring oscillators (ro_cell.v), each with three
// Configurable Inversion Units. For challenge C:
//   pair      a = C[3:0],  b = a XOR 1   (only these two run)
//   stage s of oscillator i is configured by C[(i + 5s) mod 16]
//   response  = 1 if count(a) > count(b) over WINDOW clock cycles
//
// This is the same rule as model/puf_core.py and the behavioural
// model rtl/cropuf_sim_model.v; here the frequencies come from
// real gate delays instead of a parameter table.
//
// The two edge counters are clocked by the selected oscillators.
// They are cleared before the window and read only after both
// oscillators have been stopped, so their values are static when
// the system-clock domain samples them.
//
// Interface identical to cropuf_sim_model.v.
// ============================================================
module cro_puf_core #(
    parameter integer WINDOW = 65536,     // evaluation window, clk cycles
    parameter integer CNT_W  = 24         // edge-counter width
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
    // ---------------- challenge generator: x^16+x^14+x^13+x^11+1, seed 0xACE1
    reg  [15:0] lfsr_reg;
    wire        lfsr_fb = lfsr_reg[15] ^ lfsr_reg[13] ^ lfsr_reg[12] ^ lfsr_reg[10];

    reg  [15:0] ch;                       // latched challenge
    wire [3:0]  sel_a = ch[3:0];
    wire [3:0]  sel_b = ch[3:0] ^ 4'b0001;
    reg         run;                      // oscillators enabled

    // ---------------- ring-oscillator bank
    wire [15:0] ro_out;
    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : g_ro
            wire       en_i  = run & ((sel_a == i) | (sel_b == i));
            wire [2:0] cfg_i = {ch[(i + 10) % 16], ch[(i + 5) % 16], ch[i]};
            ro_cell #(.RO(i)) u_ro (.en(en_i), .cfg(cfg_i), .ro_out(ro_out[i]));
        end
    endgenerate

    // ---------------- edge counters (clocked by the selected oscillators)
    wire clk_a = ro_out[sel_a];
    wire clk_b = ro_out[sel_b];
    reg  cnt_clr;
    reg  [CNT_W-1:0] cnt_a, cnt_b;
    always @(posedge clk_a or posedge cnt_clr)
        if (cnt_clr) cnt_a <= {CNT_W{1'b0}}; else cnt_a <= cnt_a + 1'b1;
    always @(posedge clk_b or posedge cnt_clr)
        if (cnt_clr) cnt_b <= {CNT_W{1'b0}}; else cnt_b <= cnt_b + 1'b1;

    // ---------------- control FSM (system clock)
    localparam [2:0] S_IDLE = 3'd0, S_CLR = 3'd1, S_START = 3'd2,
                     S_RUN  = 3'd3, S_SETTLE = 3'd4, S_CMP = 3'd5, S_OUT = 3'd6;
    reg [2:0]  state;
    reg [31:0] wcnt;

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE; lfsr_reg <= 16'hACE1; ch <= 16'h0000;
            run <= 1'b0; cnt_clr <= 1'b1; wcnt <= 32'd0;
            challenge_out <= 16'h0000; response_out <= 1'b0; valid <= 1'b0; busy <= 1'b0;
        end else begin
            valid <= 1'b0;
            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (auto_mode || ext_start) begin
                        if (auto_mode) begin
                            ch       <= lfsr_reg;
                            lfsr_reg <= {lfsr_reg[14:0], lfsr_fb};
                        end else
                            ch <= ext_challenge;
                        cnt_clr <= 1'b1; busy <= 1'b1; wcnt <= 32'd0;
                        state   <= S_CLR;
                    end
                end
                S_CLR: begin                      // hold the clear while the pair mux settles
                    wcnt <= wcnt + 1;
                    if (wcnt == 32'd3) begin cnt_clr <= 1'b0; state <= S_START; end
                end
                S_START: begin run <= 1'b1; wcnt <= 32'd0; state <= S_RUN; end
                S_RUN: begin
                    wcnt <= wcnt + 1;
                    if (wcnt == WINDOW - 1) begin run <= 1'b0; wcnt <= 32'd0; state <= S_SETTLE; end
                end
                S_SETTLE: begin                   // oscillators stop; counters become static
                    wcnt <= wcnt + 1;
                    if (wcnt == 32'd15) state <= S_CMP;
                end
                S_CMP: begin
                    response_out  <= (cnt_a > cnt_b) ? 1'b1 : 1'b0;
                    challenge_out <= ch;
                    state         <= S_OUT;
                end
                S_OUT: begin valid <= 1'b1; busy <= 1'b0; state <= S_IDLE; end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
