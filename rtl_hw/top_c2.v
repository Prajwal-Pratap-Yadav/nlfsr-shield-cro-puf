`timescale 1ns/1ps
// C2 - CRO-PUF + Keccak-f[200] (synthesis top)
module top_c2 #(parameter integer WINDOW = 65536) (
    input  wire        clk, rst_n, auto_mode, ext_start,
    input  wire [15:0] ext_challenge,
    output wire [15:0] challenge_out,
    output wire        obf_bit, obf_valid, busy
);
    wire raw, raw_valid;
    cro_puf_core #(.WINDOW(WINDOW)) u_cro (
        .clk(clk), .rst_n(rst_n), .auto_mode(auto_mode), .ext_challenge(ext_challenge),
        .ext_start(ext_start), .challenge_out(challenge_out), .response_out(raw),
        .valid(raw_valid), .busy(busy));
    keccak_f200_layer u_keccak (
        .clk(clk), .rst_n(rst_n), .in_bit(raw), .in_valid(raw_valid),
        .out_bit(obf_bit), .out_valid(obf_valid));
endmodule
