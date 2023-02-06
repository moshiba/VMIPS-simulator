# Instructions covered: ADD, LS
#
# Add up numbers from 1 to 10
#
# C code (-Os -mthumb -masm-syntax-unified -march=armv7-a)
# > >|int array[] = {1,2,3,4,5,6,7,8,9,10};
# > >|int acc(int* array) {
# > >|    int sum = 0;
# > >|    for (int i=0; i!=10; i+=1) {
# > >|        sum = sum + array[i];
# > >|    }
# > >|    return sum;
# > >|}
#
# Input numbers 1 to 10 are stored in scalar memory [1]-[10]
# scalar memory [0] stores 1 for initialization.
# scalar memory [11] stores 10 for end-of-loop index

LS  SR2 SR1 0    # increment step = 1
LS  SR4 SR1 11   # loop boundary = 10
LS  SR3 SR1 0    # set iterating index: i to 1
LS  SR6 SR3 0    # tmp = array[i]
ADD SR5 SR5 SR6  # sum += tmp
ADD SR3 SR3 SR2  # i+=1
BLE SR3 SR4 -4   # loop if i != boundary
HALT
