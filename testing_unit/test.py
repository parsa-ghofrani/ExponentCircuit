import struct

from mainFile import Wire, IntSqrt, MUX, FloatWire, FloatSqrt, IntMul, FloatMul, FloatSquare, Register, do_clock, \
    mainAlgorithm

# Test cases
a = Wire(0b1011, 4)
b = Wire(0b0101, 4)
print(f'{a = }')
print(f'{a[-4] = }')
print(f'{a[:3] = }')
print(f'{a - b = }')  # 5
print(f'{a + b = }')  # 15
print(f'{a.concat(b) = }')  # 0b10100101

w = Wire(0b1010, 4)
print(f'{w = }')  # Wire(value=10, bit_count=4)
w[2] = Wire(1, 1)
print(f'w after w[2] <- 1 = {w}')  # Wire(value=2, bit_count=4)

c = Wire(0b101, 3)
d = Wire(0b1, 1)
print(f'{c - d = }') # 0b100

print(f'{IntSqrt(Wire(0b1010010100000000,16)) = }') # 0b11001101
print(f'{MUX([a, b], Wire(0, 1)) = }')  # 0b1010

my_float = -1.35
"""my_float to single precision"""
# Step 1: Pack the float into bytes
packed = struct.pack('!f', my_float)

# Step 2: Unpack the bytes into an integer
unpacked = struct.unpack('!I', packed)[0]

# Step 3: Convert the integer to a binary string
binary_string = f'{unpacked:032b}'

print(f'The bits of {my_float} are: {binary_string}')

"""my_float to double precision"""
# Step 1: Pack the float into bytes (double precision)
packed = struct.pack('!d', my_float)

# Step 2: Unpack the bytes into a 64-bit integer
unpacked = struct.unpack('!Q', packed)[0]

# Step 3: Convert the integer to a 64-bit binary string
binary_string = f'{unpacked:064b}'

print(f'The bits of {my_float} are: {binary_string}')

g = FloatWire(0b10111111101011001100110011001101)
print(f'{g = }')
print(f'value of {g} = {g.to_float()}')

print(f'{Wire(0b1010, 4).SAR() = }\n')  # 0b0101

a = FloatWire(0b00111111101011001100110011001101) # 1.35
print(f'{a = } = {a.to_float()}')
print(f'{FloatSqrt(a) = } = {FloatSqrt(a)[1].to_float()}\n')  # 1.16

b = FloatWire(0b01000001100000000000000000000000) # 16.0
print(f'{b = } = {b.to_float()}')
print(f'{FloatSqrt(b) = } = {FloatSqrt(b)[1].to_float()}\n')  # 4.0

c = FloatWire(0b00111111100000000000000000000000) # 1.0
print(f'{c = } = {c.to_float()}')
print(f'{FloatSqrt(c) = } = {FloatSqrt(c)[1].to_float()}\n')  # 1.0

d = FloatWire(0b01000000000000000000000000000000) # 2.0
print(f'{d = } = {d.to_float()}')
print(f'{FloatSqrt(d) = } = {FloatSqrt(d)[1].to_float()}\n')  # 1.41

a = Wire(0b1011, 4)
b = Wire(0b0101, 4)
print(f'{IntMul(a, b) = }\n') # 0b00110111

a = FloatWire(0b00111111101011001100110011001101) # 1.35
b = FloatWire(0b01000001100000000000000000000000) # 16.0
print(f'{FloatMul(a, b) = } = {FloatMul(a, b)[2].to_float()}\n') # 0b00111111101011001100110011001101

a = FloatWire(0b00111111101011001100110011001101) # 1.35
print(f'{FloatSquare(a) = } = {FloatSquare(a)[2].to_float()}') # 1.8224999999999998


# Test Registers
a = Register(Wire(0, 4), lambda: Wire(0b1011, 4), lambda: Wire(1, 1))
b = Register(Wire(0, 4), lambda: a.value, lambda: Wire(1, 1))
print(f'{a.value = }')
print(f'{b.value = }')
do_clock()
print('Clock()')
print(f'{a.value = }')
print(f'{b.value = }')
do_clock()
print('Clock()')
print(f'{a.value = }')
print(f'{b.value = }')

# Test mainAlgorithm
# 1.35 ^ 24.0
a = FloatWire(0b00111111101011001100110011001101) # 1.35
b = FloatWire(0b01000001110000000000000000000000) # 24.0
print(f'{a.to_float() = }')
print(f'{b.to_float() = }')
# ans = mainAlgorithm(a, b)
# print(f'mainAlgorithm(a, b) = {ans} = {ans[2].to_float()}')

# 1.35 ^ 1.5
a = FloatWire(0b00111111101011001100110011001101) # 1.35
b = FloatWire(0b00111111110000000000000000000000) # 1.5
print(f'{a.to_float() = }')
print(f'{b.to_float() = }')
# ans = mainAlgorithm(a, b)
# print(f'mainAlgorithm(a, b) = {ans} = {ans[2].to_float()}')

# Test mainAlgorithm
# 1.35 ^ 24.0
a = FloatWire(0b00111111101011001100110011001101) # 1.35
b = FloatWire(0b01000001110000000000000000000000) # 24.0
print(f'{a.to_float() = }')
print(f'{b.to_float() = }')
# ans = mainAlgorithm(a, b)
# print(f'mainAlgorithm(a, b) = {ans} = {ans[2].to_float()}')

# 1.35 ^ 1.5
a = FloatWire(0b00111111101011001100110011001101) # 1.35
b = FloatWire(0b00111111110000000000000000000000) # 1.5
print(f'{a.to_float() = }')
print(f'{b.to_float() = }')
# ans = mainAlgorithm(a, b)
# print(f'mainAlgorithm(a, b) = {ans} = {ans[2].to_float()}')

# (2^128) ^ (2^-24)
a = FloatWire(0b01111111011111111111111111111111) # 1.999... * 2^127
b = FloatWire(0b00110011100000000000000000000001) # 2^-24
print(f'{a.to_float() = }')
print(f'{b.to_float() = }')
print(f'a.exponent = {(a.exponent() - a.bias)[1].twos_complement_value()}')
print(f'b.exponent = {(b.exponent() - b.bias)[1].twos_complement_value()}')
ans = mainAlgorithm(a, b)
print(f'mainAlgorithm(a, b) = {ans} = {ans[2].to_float()}')

# 1.5 ^ 2^24
a = FloatWire(0b00111111100000000000000000000001) # 1.00001
b = FloatWire(0b01001011100000000000000000000000) # 2^24
print(f'{a.to_float() = }')
print(f'{b.to_float() = }')
ans = mainAlgorithm(a, b)
print(f'mainAlgorithm(a, b) = {ans} = {ans[2].to_float()}')
