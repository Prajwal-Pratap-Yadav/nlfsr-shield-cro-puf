M=0xFFFFFFFF
def step(s):
    lin=((s>>31)^(s>>21)^(s>>1)^s)&1
    a,b,c=(s>>31)&1,(s>>15)&1,(s>>7)&1
    mnl=(a&b)^(b&c)^(a&c)
    return ((s<<1)|(lin^mnl))&M

# 1. fixed points
for s in (0x00000000,0xFFFFFFFF):
    print("fixed point check 0x%08X -> 0x%08X  %s"%(s,step(s),"YES" if step(s)==s else "no"))

# 2. independent cycle find from the paper's own seeding example
def find_cycle(s0):
    seen={}; s=s0; i=0
    while s not in seen:
        seen[s]=i; s=step(s); i+=1
    return seen[s], i-seen[s]     # tail, cycle length

for seed in (0x0000BEEF,0xACE1ACE1,0x12345678,0xDEADBEEF,1,0x7FFFFFFF):
    mu,lam=find_cycle(seed)
    print("seed 0x%08X : tail=%6d  period=%d"%(seed,mu,lam))

# 3. is the 1832-cycle unique? sample widely
import random
random.seed(7)
from collections import Counter
cnt=Counter()
for _ in range(300):
    mu,lam=find_cycle(random.getrandbits(32))
    cnt[lam]+=1
print("\nperiods over 300 random seeds:", dict(cnt))

# 4. keystream repeat: the XOR mask an attacker sees
s=0x0000BEEF
ks=[]
for _ in range(30000):
    s=step(s); ks.append(s&1)
tail=20000
seq=ks[tail:]
for p in range(1,4000):
    if all(seq[i]==seq[i+p] for i in range(2000)):
        print("keystream repeats with period %d after the transient"%p); break

# 5. non-bijectivity: exact count over a full 2^16 slice of (s15,s7)
coll=sum(1 for s15 in (0,1) for s7 in (0,1) if (1^s15^s7)==0)
print("\ncoefficient of s31 vanishes for %d of 4 (s15,s7) combinations = %.0f%% of states"%(coll,100*coll/4))
