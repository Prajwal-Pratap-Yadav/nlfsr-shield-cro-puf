`timescale 1ns/1ps
// C0 - unprotected CRO-PUF (synthesis top)
module top_c0 #(parameter integer WINDOW = 65536) (
    input  wire        clk, rst_n, auto_mode, ext_start,
    input  wire [15:0] ext_challenge,
    output wire [15:0] challenge_out,
    output wire        response_out, valid, busy
);
    cro_puf_core #(.WINDOW(WINDOW)) u_cro (
        .clk(clk), .rst_n(rst_n), .auto_mode(auto_mode), .ext_challenge(ext_challenge),
        .ext_start(ext_start), .challenge_out(challenge_out), .response_out(response_out),
        .valid(valid), .busy(busy));
endmodule
