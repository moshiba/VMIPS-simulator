# Instructions covered: LV, SV
#
# Load values from memory to register
# Store values from register to memory
#
# C code (none)
#
# Input numbers are stored in vector memory
# vmem[0]  - vmem[63] : 0-63
# vmem[64] - vmem[127]: 0

LV  VR1 SR1     # VR1 = mem[0-63]
LS  SR2 SR1  0  # SR2 = 64
SV  VR1 SR2     # mem[64-127] = VR1
HALT
