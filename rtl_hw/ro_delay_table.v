`timescale 1ps/1ps
// ============================================================
// ro_delay_table.v - SIMULATION ONLY (never synthesised)
//
// Gate delays in ps for the ring oscillators of one virtual
// device, written by tools/export_ro_delays.py from the device
// parameters of model/puf_core.py:
//   d_nand[ro]          = 20000 - base[ro]
//   d_xor [ro*3+stage]  =  8000 - delta[ro][stage][1]
//   d_xnor[ro*3+stage]  =  8000 - delta[ro][stage][0]
// A larger model value means a shorter delay, i.e. a faster
// oscillator, so  count_a > count_b  <=>  f_model(a) > f_model(b)
// up to the resolution of the counting window.
// ============================================================
module ro_delay_table;
    reg [31:0] d_nand [0:15];
    reg [31:0] d_xor  [0:47];
    reg [31:0] d_xnor [0:47];
    initial begin
        $readmemh("ro_d_nand.hex", d_nand);
        $readmemh("ro_d_xor.hex",  d_xor);
        $readmemh("ro_d_xnor.hex", d_xnor);
    end
endmodule
