`timescale 1ps/1ps
// ============================================================
// ciu.v - Configurable Inversion Unit (gate level, Fig. 2)
//
// x : ring signal from the previous stage
// s : challenge (configuration) bit
// y : output to the next stage; y = NOT x for both values of s
//
//   s = 1 : y = XOR(x, 1)  through the XOR gate
//   s = 0 : y = XNOR(x, 0) through the XNOR gate
//
// The output is the same, but the inversion is performed by a
// physically different gate, so s selects the stage delay.
//
// Synthesis: each gate is an explicitly instantiated LUT with
// DONT_TOUCH, so the tool cannot collapse the unit into a single
// inverter (which would make s inert).
// Simulation with +define+RO_TIMING_SIM: behavioural gates whose
// delays (ps) are read from ro_delay_table, so the ring really
// oscillates in an event-driven simulator.
// ============================================================
module ciu #(
    parameter integer IDX = 0          // ro*3 + stage, index into the delay table
)(
    input  wire x,
    input  wire s,
    output wire y
);
`ifdef RO_TIMING_SIM
    localparam integer D_MUX = 300;    // ps
    reg p_xor = 1'b0, p_xnor = 1'b0, y_r = 1'b0;
    always @(x or s) p_xor  <= #(ro_delay_table.d_xor[IDX])  (x ^ s);
    always @(x or s) p_xnor <= #(ro_delay_table.d_xnor[IDX]) ~(x ^ s);
    always @(p_xor or p_xnor or s) y_r <= #(D_MUX) (s ? p_xor : p_xnor);
    assign y = y_r;
`else
    (* DONT_TOUCH = "TRUE" *) wire p_xor;
    (* DONT_TOUCH = "TRUE" *) wire p_xnor;
    (* DONT_TOUCH = "TRUE" *) LUT2 #(.INIT(4'h6))  u_xor  (.O(p_xor),  .I0(x), .I1(s));
    (* DONT_TOUCH = "TRUE" *) LUT2 #(.INIT(4'h9))  u_xnor (.O(p_xnor), .I0(x), .I1(s));
    (* DONT_TOUCH = "TRUE" *) LUT3 #(.INIT(8'hCA)) u_mux  (.O(y), .I0(p_xnor), .I1(p_xor), .I2(s));
`endif
endmodule
