# Instructions covered: ADD, SUB
#
# calculate:
#     1-2+3-4+5-6+7-8+9  =  5
#     -1+2-3+4-5+6-7+8-9 = -5
#
# C code (none)
#
# Input numbers 1 to 10 are stored in scalar memory [0]-[8]
# SR0: $zero
# SR1: $one, for loop pointer increment
# SR2: data pointer
# SR3: temp data
# SR4: accumulating sum
# SR5: loop teminate condition: boundary = pointer<9
# SR6: result for add-test
# SR7: result for subtract-test

# Test add 1, -2, 3, -4, 5, -6, 7, -8, 9
LS  SR5 SR0 8    # load loop boundary: ptr=9
LS  SR1 SR0 0    # load pointer increment size: 1
LS  SR3 SR2 0    # load data into temp value holder
ADD SR4 SR4 SR3  # add temp to sum
ADD SR2 SR2 SR1  # pointer += 1
BLT SR2 SR5 -3   # loop if ptr<boundary
ADD SR6 SR4 SR0  # store result in SR6

# Reset data-pointer, temp-data, accumulating-sum registers to zero
ADD SR2 SR0 SR0
ADD SR3 SR0 SR0
ADD SR4 SR0 SR0

# Test subtract 1, -2, 3, -4, 5, -6, 7, -8, 9
LS  SR5 SR0 8    # load loop boundary: ptr=9
LS  SR1 SR0 0    # load pointer increment size: 1
LS  SR3 SR2 0    # load data into temp value holder
SUB SR4 SR4 SR3  # subtract temp from sum
ADD SR2 SR2 SR1  # pointer += 1
BLT SR2 SR5 -3   # loop if ptr<boundary
ADD SR7 SR4 SR0  # store result in SR7

HALT
