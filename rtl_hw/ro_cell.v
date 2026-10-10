`timescale 1ps/1ps
// ============================================================
// ro_cell.v - one configurable ring oscillator
//
// Five inverting stages: an enable NAND, three Configurable
// Inversion Units and one inverter (5 inversions, odd, so the
// ring oscillates for every configuration when en = 1).
// When en = 0 the NAND output is forced to 1 and the ring stops
// with ro_out = 1.
// ============================================================
module ro_cell #(
    parameter integer RO = 0
)(
    input  wire       en,
    input  wire [2:0] cfg,       // challenge bits of stages 0, 1, 2
    output wire       ro_out
);
    wire n_out, c0, c1, c2;
    (* ALLOW_COMBINATORIAL_LOOPS = "TRUE", DONT_TOUCH = "TRUE" *) wire fb;

`ifdef RO_TIMING_SIM
    localparam integer D_INV = 300;   // ps
    reg n_r = 1'b1, fb_r = 1'b1;
    always @(en or fb) n_r  <= #(ro_delay_table.d_nand[RO]) ~(en & fb);
    always @(c2)       fb_r <= #(D_INV) ~c2;
    assign n_out = n_r;
    assign fb    = fb_r;
`else
    (* DONT_TOUCH = "TRUE" *) LUT2 #(.INIT(4'h7)) u_nand (.O(n_out), .I0(fb), .I1(en));
    (* DONT_TOUCH = "TRUE" *) LUT1 #(.INIT(2'h1)) u_inv  (.O(fb),    .I0(c2));
`endif

    ciu #(.IDX(RO*3 + 0)) u_s0 (.x(n_out), .s(cfg[0]), .y(c0));
    ciu #(.IDX(RO*3 + 1)) u_s1 (.x(c0),    .s(cfg[1]), .y(c1));
    ciu #(.IDX(RO*3 + 2)) u_s2 (.x(c1),    .s(cfg[2]), .y(c2));

    assign ro_out = fb;
endmodule
