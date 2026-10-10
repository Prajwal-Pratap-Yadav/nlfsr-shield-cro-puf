# ============================================================
# export_ro_delays.py
#
# Writes the gate delays (ps) of one virtual device for the
# timing simulation of the real ring-oscillator RTL (rtl_hw/,
# compiled with +define+RO_TIMING_SIM):
#   ro_d_nand.hex (16), ro_d_xor.hex (48), ro_d_xnor.hex (48)
#
#   python tools/export_ro_delays.py [device_id]     (default 0)
#
# Mapping (see rtl_hw/ro_delay_table.v):
#   d_nand[ro]         = 20000 - base[ro]
#   d_xor [ro*3+s]     =  8000 - delta[ro][s][1]
#   d_xnor[ro*3+s]     =  8000 - delta[ro][s][0]
# ============================================================
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
from puf_core import build_device, NUM_ROS, STAGES

D_NAND0, D_STAGE0 = 20000, 8000
dev_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
base, delta = build_device(dev_id)
with open("ro_d_nand.hex", "w") as f:
    for ro in range(NUM_ROS):
        f.write("%08X\n" % (D_NAND0 - int(base[ro])))
with open("ro_d_xor.hex", "w") as fx, open("ro_d_xnor.hex", "w") as fn:
    for ro in range(NUM_ROS):
        for s in range(STAGES):
            fx.write("%08X\n" % (D_STAGE0 - int(delta[ro, s, 1])))
            fn.write("%08X\n" % (D_STAGE0 - int(delta[ro, s, 0])))
print("device %d delays written (ro_d_nand.hex, ro_d_xor.hex, ro_d_xnor.hex)" % dev_id)
