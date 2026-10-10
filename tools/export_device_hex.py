# ============================================================
# export_device_hex.py
#
# Writes the parameters of virtual device(s) as hex files for
# $readmemh in rtl/cropuf_sim_model.v (version 3.0).
#
#   python tools/export_device_hex.py            # device 0
#   python tools/export_device_hex.py 0 1 2      # devices 0, 1, 2
#
# Output: device_NNN_base.hex (16 words) and
#         device_NNN_delta.hex (96 words) in the current folder.
# ============================================================
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
from puf_core import build_device, write_device_hex

ids = [int(a) for a in sys.argv[1:]] or [0]
for d in ids:
    b, t = "device_%03d_base.hex" % d, "device_%03d_delta.hex" % d
    write_device_hex(build_device(d), b, t)
    print("device %d -> %s, %s" % (d, b, t))
