`timescale 1ns/1ps
// Timing simulation of top_c1 (real RO bank + NLFSR-Shield).
// Dumps every raw response (from inside the core) and every
// obfuscated output, so tools/check_c1_hw.py can verify
// z = r XOR k against the Python NLFSR seeded with the same raw bits.
module tb_c1_timing;
    parameter integer N = 400;
    reg clk = 0, rst_n = 0, auto_mode = 0, ext_start = 0;
    reg [15:0] ext_challenge = 16'h0;
    wire [15:0] challenge_out; wire response_out, valid, busy, ready;
    integer fr, fz, n;
    top_c1 #(.WINDOW(20000)) uut (
        .clk(clk), .rst_n(rst_n), .auto_mode(auto_mode), .ext_start(ext_start),
        .ext_challenge(ext_challenge), .challenge_out(challenge_out),
        .response_out(response_out), .valid(valid), .busy(busy), .ready(ready));
    always #5 clk = ~clk;
    always @(negedge clk) if (uut.raw_valid) $fwrite(fr, "%04x %b\n", uut.ch_raw, uut.raw);
    initial begin
        fr = $fopen("c1_raw.txt", "w"); fz = $fopen("c1_out.txt", "w"); n = 0;
        repeat (4) @(negedge clk);
        rst_n = 1; @(negedge clk);
        auto_mode = 1;
        while (n < N) begin
            @(negedge clk);
            if (valid) begin $fwrite(fz, "%04x %b\n", challenge_out, response_out); n = n + 1; end
        end
        $fclose(fr); $fclose(fz); $display("C1: wrote %0d outputs", n); $finish;
    end
endmodule
