## System clock: 100 MHz (10 ns), as in the paper.
create_clock -period 10.000 -name clk [get_ports clk]
## The ring-oscillator loops are intentional; run_all.tcl marks their
## nets with ALLOW_COMBINATORIAL_LOOPS after synthesis.
