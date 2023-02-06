# Instructions covered: ADDVV,SUBVV
#
# Vectorized add / subtract
#
# C code (none)
#
# Input numbers are stored in vector memory
# vmem[0]  - vmem[63] : 0-63
# vmem[64] - vmem[127]: 1-64

LV    VR1 SR1     # VR1 = mem[0-63]
LS    SR2 SR1 0   # SR2 = 64
LV    VR2 SR2     # VR2 = mem[64-127]
SUBVV VR3 VR2 VR1 # VR3 = VR2 - VR1
LS    SR2 SR1 1   # SR2 = 128
SV    VR3 SR2     # mem[128-191] = VR1
HALT
