LS   SR1 SR0 0    # Loads 128 into SR1
LS   SR2 SR0 1    # Loads 64 into SR2
LS   SR4 SR0 2    # Loads 2 into SR4
ADD  SR0 SR0 SR2  # Increments SR0 by 64
LV   VR0 SR3      # Loads 64 elements into VR0 starting from address 0 with stride 1.
LVWS VR1 SR3 SR4  # Loads 64 elements into VR1 starting from address 0 with stride 2.
LVI  VR2 SR3 VR0  # Gathers 64 elements into VR2 with base address 0 and offsets in VR0.
BLT  SR0 SR1 -4   # Branch to PC 3 till SR0 is 128
HALT
