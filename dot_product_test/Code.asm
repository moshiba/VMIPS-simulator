# dot procuct
#
# vec[450] X vec[450]
#
# SR1: $zero
# SR2: pointer
# SR3: pointer increment step
# SR4: strip mining short piece size
# SR5: loop size
# VR1: tmp vector 1
# VR2: tmp vector 2
# VR3: sum vector

LS    SR3 SR1 0    # get pointer increment step = 64
LS    SR5 SR1 2    # get loop termination condition

# Strip mining
# - 1. short piece
LS    SR4 SR1 1    # get strip mining short piece size = 450 mod 64 = 2
MTCL  SR4          # set vector_length
#
LV    VR1 SR2      # load vector starting from address [SR2]
LV    VR2 SR2      # load vector starting from address [SR2]
MULVV VR1 VR1 VR2  # multiply and save product to tmp1
ADDVV VR3 VR3 VR1  # sum += tmp
#
MTCL  SR3          # reset vec_length = MVL
ADD   SR2 SR1 SR4  # update pointer

# - 2. whole vector
LV    VR1 SR2      # load vector starting from address [SR2]
LV    VR2 SR2      # load vector starting from address [SR2]
MULVV VR1 VR1 VR2  # multiply and save product to tmp1
ADDVV VR3 VR3 VR1  # sum += tmp
ADD   SR2 SR2 SR3  # pointer += step
BNE   SR2 SR5 -6   # loop, if pointer < element_size

# Gather result
HALT
