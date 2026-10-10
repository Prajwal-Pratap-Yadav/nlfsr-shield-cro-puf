`timescale 1ns/1ps
// C1 - CRO-PUF + NLFSR-Shield (synthesis top)
// The first 32 raw responses seed the NLFSR (no output).
// Afterwards every raw response r is emitted as z = r XOR k, where
// k is the current NLFSR output, and the NLFSR advances by one step.
module top_c1 #(parameter integer WINDOW = 65536) (
    input  wire        clk, rst_n, auto_mode, ext_start,
    input  wire [15:0] ext_challenge,
    output reg  [15:0] challenge_out,
    output reg         response_out,
    output reg         valid,
    output wire        busy,
    output wire        ready
);
    wire [15:0] ch_raw; wire raw, raw_valid, k;
    cro_puf_core #(.WINDOW(WINDOW)) u_cro (
        .clk(clk), .rst_n(rst_n), .auto_mode(auto_mode), .ext_challenge(ext_challenge),
        .ext_start(ext_start), .challenge_out(ch_raw), .response_out(raw),
        .valid(raw_valid), .busy(busy));
    nlfsr_core u_nlfsr (
        .clk(clk), .rst_n(rst_n), .seed_mode(raw_valid & ~ready), .seed_bit(raw),
        .advance(raw_valid & ready), .nlfsr_out(k), .ready(ready));
    always @(posedge clk) begin
        if (!rst_n) begin
            challenge_out <= 16'h0000; response_out <= 1'b0; valid <= 1'b0;
        end else begin
            valid <= 1'b0;
            if (raw_valid && ready) begin
                challenge_out <= ch_raw; response_out <= raw ^ k; valid <= 1'b1;
            end
        end
    end
endmodule
