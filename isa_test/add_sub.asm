# Instructions covered: ADD, SUB
#
# calculate:
#     1-2+3-4+5-6+7-8+9  =  5
#     -1+2-3+4-5+6-7+8-9 = -5
#
# C code (none)
#
# Input numbers 1 to 10 are stored in scalar memory [0]-[8]
# SR1: $zero
# SR2: $one, for loop pointer increment
# SR3: data pointer
# SR4: temp data
# SR5: accumulating sum
# SR6: loop teminate condition: boundary = pointer<9
# SR7: result for add-test
# SR8: result for subtract-test

# Test add 1, -2, 3, -4, 5, -6, 7, -8, 9
LS  SR6 SR1 8    # load loop boundary: ptr=9
LS  SR2 SR1 0    # load pointer increment size: 1
LS  SR4 SR3 0    # load data into temp value holder
ADD SR5 SR5 SR4  # add temp to sum
ADD SR3 SR3 SR2  # pointer += 1
BLT SR3 SR6 -3   # loop if ptr<boundary
ADD SR7 SR5 SR1  # store result in SR7

# Reset data-pointer, temp-data, accumulating-sum registers to zero
ADD SR3 SR1 SR1
ADD SR4 SR1 SR1
ADD SR5 SR1 SR1

# Test subtract 1, -2, 3, -4, 5, -6, 7, -8, 9
LS  SR6 SR1 8    # load loop boundary: ptr=9
LS  SR2 SR1 0    # load pointer increment size: 1
LS  SR4 SR3 0    # load data into temp value holder
SUB SR5 SR5 SR4  # subtract temp from sum
ADD SR3 SR3 SR2  # pointer += 1
BLT SR3 SR6 -3   # loop if ptr<boundary
ADD SR8 SR5 SR1  # store result in SR8

HALT
