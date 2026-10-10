Reports of the run used for Table VII (Vivado 2025.2, xc7a35tcpg236-1, 10 ns clock).

summary.txt was written by the first version of run_all.tcl. Its column
"LUTs in RO bank" (200) counted LUT cells by hierarchical name and also matched
LUTs other than the ring-oscillator gates, so the column "outside RO bank" is
not meaningful. The ring oscillators consist of 176 LUT primitives
(16 rings x 11 gates, all DONT_TOUCH); every configuration contains exactly 16
combinational loops (check_timing "loops", DRC LUTLP-2). Slice LUTs (201/218/597)
are after LUT combining: 297 LUT primitives occupy 201 slice LUTs in C0.
The current run_all.tcl counts the DONT_TOUCH LUTs directly.

top_c0.bit, top_c1.bit, top_c2.bit: bitstreams (pins unassigned; a board-specific
pin constraint file and a UART read-out are needed before programming).
