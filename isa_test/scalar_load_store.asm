# Instructions covered: LS, SS
#
# Load values from memory to register
# Store values from register to memory
#
# C code (none)
#
# Input numbers are some primes stored in scalar memory location [0] to [3]

# keep SR1 = 0
LS  SR2 SR1 0  # SR2 = mem[0]
LS  SR3 SR1 1  # SR3 = mem[1]
LS  SR4 SR1 2  # SR4 = mem[2]
LS  SR5 SR1 3  # SR5 = mem[3]

SS  SR2 SR1 4  # mem[4] = SR2
SS  SR3 SR1 5  # mem[5] = SR3
SS  SR4 SR1 6  # mem[6] = SR4
SS  SR5 SR1 7  # mem[7] = SR5
HALT
