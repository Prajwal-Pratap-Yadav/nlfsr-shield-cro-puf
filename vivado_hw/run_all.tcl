# ============================================================
# run_all.tcl - NLFSR-Shield: synthesis, implementation and
#               bitstream generation of C0, C1 and C2 with the
#               real ring-oscillator RTL (folder rtl_hw).
#
# HOW TO RUN (no FPGA board needed):
#   1. Start Vivado. Do NOT open or create a project.
#   2. In the Tcl Console at the bottom, type (use forward slashes):
#        cd D:/Documents/nlfsr_shield_puf/supplementary_v3/vivado_hw
#        source run_all.tcl
#   3. Wait 10-25 minutes. Results go to vivado_hw/reports/,
#      the key numbers to reports/summary.txt.
# ============================================================
set PART xc7a35tcpg236-1
set HERE [file normalize [file dirname [info script]]]
set HW   [file normalize $HERE/../rtl_hw]
set RTL  [file normalize $HERE/../rtl]
set OUT  [file normalize $HERE/reports]
file mkdir $OUT

set SRC [list $HW/ciu.v $HW/ro_cell.v $HW/cro_puf_core.v $RTL/nlfsr_core.v $HW/keccak_f200_layer.v]

proc grab {text pattern} {
    if {[regexp $pattern $text -> v]} { return $v } else { return "n/a" }
}

set sf [open $OUT/summary.txt w]
puts $sf "NLFSR-Shield post-implementation summary  ([clock format [clock seconds]])"
puts $sf "Vivado [version -short], part $PART, clock 10 ns"
puts $sf "cfg | Slice LUTs | LUT primitives in ring oscillators (DONT_TOUCH) | Slice registers | Slices | WNS ns | WHS ns | Fmax MHz | Dynamic W | Total on-chip W | Bitstream"
close $sf

foreach cfg {c0 c1 c2} {
    puts "\n================ $cfg ================"
    close_project -quiet
    create_project -in_memory -part $PART
    read_verilog $SRC
    read_verilog $HW/top_$cfg.v
    read_xdc $HERE/nlfsr_shield.xdc
    synth_design -top top_$cfg -part $PART

    # the 16 ring-oscillator loops are intentional
    set ro_nets [get_nets -quiet -hierarchical -filter {NAME =~ "*u_ro/*"}]
    if {[llength $ro_nets] > 0} { set_property ALLOW_COMBINATORIAL_LOOPS TRUE $ro_nets }

    opt_design
    place_design
    route_design

    report_utilization                -file $OUT/${cfg}_utilization.txt
    report_utilization -hierarchical  -file $OUT/${cfg}_utilization_hier.txt
    report_timing_summary -max_paths 10 -file $OUT/${cfg}_timing.txt
    report_power                      -file $OUT/${cfg}_power.txt
    report_drc                        -file $OUT/${cfg}_drc.txt
    check_timing                      -file $OUT/${cfg}_check_timing.txt

    set u [report_utilization -return_string]
    set p [report_power -return_string]
    set luts  [grab $u {\|\s*Slice LUTs\*?\s*\|\s*(\d+)}]
    set ffs   [grab $u {\|\s*Slice Registers\s*\|\s*(\d+)}]
    set slc   [grab $u {\|\s*Slice\s*\|\s*(\d+)}]
    # the ring-oscillator gates are the only DONT_TOUCH LUTs (16 x 11 = 176)
    set ro    [llength [get_cells -quiet -hierarchical -filter {PRIMITIVE_GROUP == LUT && DONT_TOUCH}]]
    set wns   [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
    set whs   [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]
    set fmax  [format %.1f [expr {1000.0 / (10.0 - $wns)}]]
    set dyn   [grab $p {Dynamic \(W\)\s*\|\s*([0-9.]+)}]
    set tot   [grab $p {Total On-Chip Power \(W\)\s*\|\s*([0-9.]+)}]

    # bitstream: pins are deliberately unassigned (no board), so the
    # two pin-assignment checks are downgraded to warnings
    catch {set_property SEVERITY {Warning} [get_drc_checks NSTD-1]}
    catch {set_property SEVERITY {Warning} [get_drc_checks UCIO-1]}
    catch {set_property BITSTREAM.GENERAL.UNCONSTRAINEDPINS Allow [current_design]}
    if {[catch {write_bitstream -force $OUT/top_$cfg.bit} err]} {
        set bit "FAILED ([lindex [split $err \n] 0])"
    } else { set bit "written" }

    set sf [open $OUT/summary.txt a]
    puts $sf "$cfg | $luts | $ro | $ffs | $slc | $wns | $whs | $fmax | $dyn | $tot | $bit"
    close $sf
    close_project
}
puts "\nDONE. Send me the file: $OUT/summary.txt (and the whole reports folder as a zip)."
