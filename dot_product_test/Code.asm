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
BNE   SR2 SR5 -5   # loop, if pointer < element_size

# Gather result
LS    SR6 SR1 3    # load loop end condition: vector_length == 1
SV    VR3 SR2      # save scattered result
SRA   SR3 SR3 SR6  # vector_length /= 2 (initially at 64)
MTCL  SR3          # set new vector_length
LV    VR3 SR2      # load previously stored data as two smaller vectors (higher half)
ADD   SR2 SR2 SR3  # update pointer to load next partition
LV    VR4 SR2      # load previously stored data as two smaller vectors (lower half)
ADD   SR2 SR2 SR3  # update pointer to next head
ADDVV VR3 VR3 VR4  # sum these two smaller vectors
BNE   SR3 SR6 -8   # loop, if vector_length > 1

# Store result to designated address
LS    SR6 SR1 4    # load target address: 2048
SV    VR3 SR6      # save final result
HALT
