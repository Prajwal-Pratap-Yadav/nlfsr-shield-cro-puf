`timescale 1ns/1ps
// ============================================================
// tb_ro_timing.v - timing simulation of the REAL ring-oscillator
// CRO-PUF (cro_puf_core.v) with gate delays of one virtual device.
//
//   python ../tools/export_ro_delays.py 0
//   iverilog -g2005 -DRO_TIMING_SIM -o sim_ro ro_delay_table.v ciu.v \
//            ro_cell.v cro_puf_core.v tb_ro_timing.v
//   vvp sim_ro              -> ro_timing_out.txt
//   python ../tools/check_ro_timing.py
//
// The oscillators really oscillate here (event-driven simulation
// with non-zero delays). The window is shortened to 20,000 clock
// cycles (200 us) to keep the run short; with ~90 ns loop periods
// this gives about 2,200 oscillations per evaluation.
// ============================================================
module tb_ro_timing;
    parameter integer N      = 2000;
    parameter integer WINDOW = 20000;
    reg clk = 0, rst_n = 0, auto_mode = 0, ext_start = 0;
    reg [15:0] ext_challenge = 16'h0;
    wire [15:0] challenge_out; wire response_out, valid, busy;
    integer f, n;
    cro_puf_core #(.WINDOW(WINDOW)) uut (
        .clk(clk), .rst_n(rst_n), .auto_mode(auto_mode), .ext_challenge(ext_challenge),
        .ext_start(ext_start), .challenge_out(challenge_out), .response_out(response_out),
        .valid(valid), .busy(busy));
    always #5 clk = ~clk;
    initial begin
        f = $fopen("ro_timing_out.txt", "w"); n = 0;
        repeat (4) @(negedge clk);
        rst_n = 1; @(negedge clk);
        auto_mode = 1;
        while (n < N) begin
            @(negedge clk);
            if (valid) begin
                $fwrite(f, "%04x %b %0d %0d\n", challenge_out, response_out, uut.cnt_a, uut.cnt_b);
                n = n + 1;
            end
        end
        $fclose(f); $display("wrote %0d CRPs", n); $finish;
    end
endmodule
