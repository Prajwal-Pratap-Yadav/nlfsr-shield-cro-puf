# Their NLFSRModel, copied verbatim from generate_c1_c2_datasets.py
class NLFSRModel:
    def __init__(self, width=32):
        self.width=width; self.state=0; self.ready=False; self.seed_count=0
    def _linear_fb(self):
        s=self.state
        return ((s>>31)^(s>>21)^(s>>1)^s)&1
    def _bent_fb(self):
        a=(self.state>>31)&1; b=(self.state>>15)&1; c=(self.state>>7)&1
        return (a&b)^(b&c)^(a&c)
    def _combined_fb(self):
        return self._linear_fb()^self._bent_fb()
    def seed_bit(self,bit):
        if self.ready: return True
        self.state=(((self.state<<1)&((1<<self.width)-1))|(bit&1))
        self.seed_count+=1
        if self.seed_count>=self.width: self.ready=True
        return self.ready
    def next_bit(self):
        out=self._combined_fb()
        self.state=(((self.state<<1)&((1<<self.width)-1))|out)
        return out

def period_from(state0):
    n=NLFSRModel(); n.state=state0; n.ready=True
    seen={}; i=0
    while n.state not in seen:
        seen[n.state]=i; n.next_bit(); i+=1
    return seen[n.state], i-seen[n.state]

# Their exact PUF model
LCG_MULT,LCG_INC=4259,47
DEVICE_SEED,LFSR_SEED=0xBEEF,0xACE1
def lfsr_next(r): return ((r<<1)|(((r>>15)^(r>>13)^(r>>12)^(r>>10))&1))&0xFFFF
def generate_fingerprints(seed):
    fps=[seed&0xFFFF]
    for i in range(1,16): fps.append((fps[-1]*LCG_MULT+LCG_INC)&0xFFFF)
    return fps
def compute_raw_response(ch,fps):
    a=ch&0xF; b=a^0x1
    fa=fps[a]+(256 if (ch>>a)&1 else 0)
    fb=fps[b]+(256 if (ch>>b)&1 else 0)
    return 1 if fa>fb else 0

def device_seeds(n):
    s=DEVICE_SEED; out=[s]
    for _ in range(n-1):
        s=(s*LCG_MULT+LCG_INC)&0xFFFF; out.append(s)
    return out

print("=== Period of the NLFSR as implemented in your generator ===")
for dseed in device_seeds(10):
    fps=generate_fingerprints(dseed)
    nl=NLFSRModel(); lf=LFSR_SEED
    for _ in range(32):
        nl.seed_bit(compute_raw_response(lf,fps)); lf=lfsr_next(lf)
    mu,lam=period_from(nl.state)
    print("  device seed 0x%04X -> seeded state 0x%08X : tail=%6d  PERIOD=%d"%(dseed,nl.state,mu,lam))

print("\n=== CIU offset effectiveness (your pair rule sel_b = sel_a XOR 1) ===")
fps=generate_fingerprints(DEVICE_SEED)
lf=LFSR_SEED; flips=0; tot=0
for _ in range(65535):
    a=lf&0xF; b=a^0x1
    base=1 if fps[a]>fps[b] else 0
    if compute_raw_response(lf,fps)!=base: flips+=1
    tot+=1; lf=lfsr_next(lf)
print("  CIU offset changes the outcome on %d of %d challenges (%.4f%%)"%(flips,tot,100*flips/tot))

print("\n=== C0 uniformity with the correct pair rule ===")
import numpy as np
allb=[]
for dseed in device_seeds(10):
    fps=generate_fingerprints(dseed); lf=LFSR_SEED
    bits=[]
    for _ in range(100000):
        bits.append(compute_raw_response(lf,fps)); lf=lfsr_next(lf)
    allb.extend(bits)
print("  pooled over 10 devices, 1M CRPs: %.4f%%"%(100*np.mean(allb)))
