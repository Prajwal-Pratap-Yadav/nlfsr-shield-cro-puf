`timescale 1ns/1ps
// Equivalence testbench for keccak_f200_layer.v: feeds the raw bits
// in keccak_in.txt (one 0/1 per line) with gaps between them, as
// the PUF core would, and writes every output bit to keccak_out.txt.
module tb_keccak_layer;
    parameter integer NBITS = 640;              // 10 blocks of 64
    reg clk = 0, rst_n = 0, in_bit = 0, in_valid = 0;
    wire out_bit, out_valid;
    reg  bits [0:NBITS-1];
    integer f, i, nout;
    keccak_f200_layer uut (.clk(clk), .rst_n(rst_n), .in_bit(in_bit), .in_valid(in_valid),
                           .out_bit(out_bit), .out_valid(out_valid));
    always #5 clk = ~clk;
    always @(negedge clk) if (out_valid) begin $fwrite(f, "%b\n", out_bit); nout = nout + 1; end
    initial begin
        $readmemb("keccak_in.txt", bits);
        f = $fopen("keccak_out.txt", "w"); nout = 0;
        repeat (3) @(negedge clk); rst_n = 1;
        for (i = 0; i < NBITS; i = i + 1) begin
            @(negedge clk); in_bit = bits[i]; in_valid = 1;
            @(negedge clk); in_valid = 0;
            repeat (100) @(negedge clk);           // a real CRP takes ~65,540 cycles
        end
        repeat (200) @(negedge clk);
        $fclose(f); $display("keccak layer: %0d output bits", nout); $finish;
    end
endmodule
