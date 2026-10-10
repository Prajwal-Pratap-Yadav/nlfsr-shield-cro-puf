`timescale 1ns/1ps
// Equivalence testbench: dumps all 65,535 CRPs of device 0 (one full LFSR period) from
// cropuf_sim_model.v (v3.0) for comparison with model/puf_core.py.
// Run tools/export_device_hex.py first to create the hex files.
// Stimulus is applied on the negative clock edge (see README).
module tb_check;
  reg clk=0, rst_n=0, auto_mode=0, ext_start=0;
  reg [15:0] ext_challenge=0;
  wire [15:0] challenge_out; wire response_out, valid, busy;
  integer f, n;
  cropuf_sim_model #(.BASE_FILE("device_000_base.hex"),
                     .DELTA_FILE("device_000_delta.hex")) uut (
    .clk(clk), .rst_n(rst_n), .auto_mode(auto_mode),
    .ext_challenge(ext_challenge), .ext_start(ext_start),
    .challenge_out(challenge_out), .response_out(response_out),
    .valid(valid), .busy(busy));
  always #5 clk = ~clk;
  initial begin
    f = $fopen("rtl_out.txt","w"); n = 0;
    repeat(4) @(negedge clk);
    rst_n = 1; @(negedge clk);
    auto_mode = 1;
    while (n < 65535) begin
      @(negedge clk);
      if (valid) begin
        $fwrite(f, "%04x %b\n", challenge_out, response_out);
        n = n + 1;
      end
    end
    $fclose(f); $display("RTL wrote %0d CRPs", n); $finish;
  end
endmodule
