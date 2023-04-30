# Fully Connected Layer
#
# vector[256] X matrix[256x256]
#
# <CONSTANTS>
# SR0: $zero
# SR1: 1
# SR2: loop iteration=256
# SR3: vector size=64
# SR4: segment stride=256*64=16384
#
# <VARIABLES>
# SR5: result address
# SR6: input element pointer
# SR7: weight element pointer
#
# VR0: input elements  (i0, i0, i0, ...)
# VR1: weight elements (A0, B0, C0, D0, ...) segment 1, or segment 3
# VR2: weight elements (E0, F0, G0, H0, ...) segment 2, or segment 4
#
# VR4: intermediate result segment 1
# VR5: intermediate result segment 2
# VR6: intermediate result segment 3
# VR7: intermediate result segment 4
#

# Init variables
LS    SR1 SR0 0    # [const] 1
LS    SR2 SR0 1    # [const] 256
LS    SR3 SR0 2    # [const] 64
LS    SR4 SR0 3    # [const] 256*64=16384
LS    SR5 SR0 4    # [var]   256*(256+1)=65792
ADD   SR7 SR6 SR2  # [var]   (input element pointer)+256

# Loop
LVWS  VR0 SR6 SR0  # load input elements
LVWS  VR1 SR7 SR2  # load weight elements segment 1
ADD   SR7 SR7 SR4  # weight elements pointer += segment stride
LVWS  VR2 SR7 SR2  # load weight elements segment 2
MULVV VR1 VR1 VR0  # multiply segment 1
ADDVV VR4 VR4 VR1  # accumulate segment 1
MULVV VR2 VR2 VR0  # multiply segment 2
ADDVV VR5 VR5 VR2  # accumulate segment 2
ADD   SR7 SR7 SR4  # weight elements pointer += segment stride
LVWS  VR1 SR7 SR2  # load weight elements segment 3
ADD   SR7 SR7 SR4  # weight elements pointer += segment stride
LVWS  VR2 SR7 SR2  # load weight elements segment 4
MULVV VR1 VR1 VR0  # multiply segment 3
ADDVV VR6 VR6 VR1  # accumulate segment 3
MULVV VR2 VR2 VR0  # multiply segment 4
ADDVV VR7 VR7 VR2  # accumulate segment 4
ADD   SR6 SR6 SR1  # input pointer +=1
ADD   SR7 SR6 SR2  # reset weight element pointer for next iteration
BNE   SR6 SR2 -18  # loop if pointer < vector-dimension

# Store result to designated address
SV    VR4 SR5      # save final result segment 1
ADD   SR5 SR5 SR3  # set target address for segment 2
SV    VR5 SR5      # save final result segment 2
ADD   SR5 SR5 SR3  # set target address for segment 3
SV    VR6 SR5      # save final result segment 3
ADD   SR5 SR5 SR3  # set target address for segment 4
SV    VR7 SR5      # save final result segment 4
HALT
