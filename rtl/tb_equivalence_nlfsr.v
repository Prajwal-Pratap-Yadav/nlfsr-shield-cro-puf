`timescale 1ns/1ps
module tb_nlfsr_check;
  reg clk=0, rst_n=0, seed_mode=0, seed_bit=0;
  wire nlfsr_out, ready;
  integer f, n; reg [31:0] seedvec;
  nlfsr_core uut(.clk(clk),.rst_n(rst_n),.seed_mode(seed_mode),
                 .seed_bit(seed_bit),.advance(1'b1),.nlfsr_out(nlfsr_out),.ready(ready));
  always #5 clk=~clk;
  initial begin
    f=$fopen("nlfsr_out.txt","w");
    seedvec = 32'h50C04CB1;
    @(negedge clk); rst_n=0;
    @(negedge clk); rst_n=1;
    seed_mode=1;
    for (n=31; n>=0; n=n-1) begin
      seed_bit = seedvec[n];
      @(negedge clk);
    end
    seed_mode=0;
    $display("state after seeding = %h  ready=%b", uut.state, ready);
    for (n=0; n<3000; n=n+1) begin
      $fwrite(f,"%b", nlfsr_out);
      @(negedge clk);
    end
    $fclose(f); $display("NLFSR wrote 3000 bits"); $finish;
  end
endmodule
