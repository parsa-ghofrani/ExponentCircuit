from typing import List
import struct
# defining wires
class Wire:
    value: int
    bit_count: int

    def __init__(self, value: int, bit_count: int) -> None:
        self.value = value
        self.bit_count = bit_count
    
    def __eq__(self, other):
        if isinstance(other, Wire):
            max_bit_count = max(self.bit_count, other.bit_count)
            mask = (1 << max_bit_count) - 1  # Create a mask to compare only the first n bits
            return (self.value & mask) == (other.value & mask)
        else:
            mask = (1 << self.bit_count) - 1
            return (self.value & mask) == (other & mask)

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0:
                key += self.bit_count
            return Wire((self.value >> key) & 1, 1)
        elif isinstance(key, slice): # TODO: implement slice for negative values
            start = key.start or 0
            stop = key.stop or (self.bit_count + 1)
            if start < 0:
                start += self.bit_count
            if stop < 0:
                stop += self.bit_count
            mask = (1 << (stop - start)) - 1
            return Wire((self.value >> start) & mask, stop - start)
    

    def __setitem__(self, key, value):
        if not isinstance(value, Wire):
            raise ValueError("Value must be a Wire instance")
        if isinstance(key, int):
            if key < 0:
                key += self.bit_count
            if key < 0 or key >= self.bit_count:
                raise IndexError("Wire index out of range")
            if value.value == 1:
                self.value |= (1 << key)
            else:
                self.value &= ~(1 << key)
        elif isinstance(key, slice):
            start = key.start or 0
            stop = key.stop or (self.bit_count + 1)
            mask = (1 << (stop - start)) - 1
            self.value = (self.value & ~(mask << start)) | (value.value << start)
        else:
            raise TypeError("Invalid argument type")
    
    def __and__(self, other):
        if isinstance(other, Wire):
            return Wire(self.value & other.value, max(self.bit_count, other.bit_count))
        else:
            return Wire(self.value & other, self.bit_count)
    
    def __xor__(self, other):
        if isinstance(other, Wire):
            return Wire(self.value ^ other.value, max(self.bit_count, other.bit_count))
        else:
            return Wire(self.value ^ other, self.bit_count)

    def __or__(self, other):
        if isinstance(other, Wire):
            return Wire(self.value | other.value, max(self.bit_count, other.bit_count))
        else:
            return Wire(self.value | other, self.bit_count)

    def __invert__(self):
        # Create a mask with all bits set within the bit_count
        mask = (1 << self.bit_count) - 1
        # XOR the value with the mask to flip all bits
        inverted_value = self.value ^ mask
        # Return a new Wire instance with the inverted value
        return Wire(inverted_value, self.bit_count)
    
    def __lshift__(self, other):
        return Wire(self.value << other, self.bit_count + other)
    
    def __rshift__(self, other):
        return Wire(self.value >> other, self.bit_count - other)
    
    def __repr__(self):
        return f"0b{self.value:0{self.bit_count}b}"
    
    def __add__(self, other):
        a = self
        b = other if isinstance(other, Wire) else Wire(other, a.bit_count)
        n = max(a.bit_count, b.bit_count)
        result = Wire(0, n)
        carry = Wire(0, 1)

        for i in range(n):
            # Full adder logic
            result |= (a[i] ^ b[i] ^ carry) << i
            carry = (a[i] & b[i]) | (b[i] & carry) | (a[i] & carry)

        return carry, result
    
    def __sub__(self, other):
        a = self
        b = other if isinstance(other, Wire) else Wire(other, a.bit_count)
        n = max(a.bit_count, b.bit_count)
        if a.bit_count < b.bit_count:
            a = Wire(0, b.bit_count - a.bit_count) // a
        elif b.bit_count < a.bit_count:
            b = Wire(0, a.bit_count - b.bit_count) // b
        result = Wire(0, n)
        borrow = Wire(0, 1)

        for i in range(n):
            # Full subtractor logic
            result |= (a[i] ^ b[i] ^ borrow) << i
            borrow = borrow & ~(a[i] ^ b[i]) | ~a[i] & b[i]

        return borrow, result
    
    def __mul__(self, other: int):
        if isinstance(other, int): # concat n times
            result = self
            for _ in range(other - 1):
                result = result.concat(self)
            return result
        else:
            return IntMul(self, other)

    def concat(self, other):
        return Wire((self.value << other.bit_count) | other.value, self.bit_count + other.bit_count)

    def __floordiv__(self, other): # // is concatenation
        return self.concat(other)

    def reverse(self):
        value = self.value
        binary = bin(value)[2:]
        reversed_bin = binary[::-1]
        num_of_zeros = self.bit_count - len(reversed_bin)
        reversed_bin += '0' * num_of_zeros
        reversed_bin = int(reversed_bin, 2)
        return Wire(reversed_bin, self.bit_count)
    
    def multi_bit_nor(self):
        for i in range(self.bit_count):
            if self[i].value == 1:
                return Wire(0, 1)
        return Wire(1, 1)
    
    def SAR(self):
        is_negative = self[-1]
        result = self[-1] // self
        return result >> 1
    
class FloatWire(Wire):
    f: int
    e: int
    bias: Wire

    def __init__(self, value: int, e: int = 8, f: int = 23, bias: Wire = Wire(127,8)) -> None:
        Wire.__init__(self, value, e + f + 1)
        self.e = e
        self.f = f
        self.bias = bias
    
    # def __init__(self, wire: Wire, e: int = 8, f: int = 23, bias: Wire = Wire(127,8)) -> None:
    #     Wire.__init__(self, wire.value, e + f + 1)
    #     self.e = e
    #     self.f = f
    #     self.bias = bias
    
    def __repr__(self):
        return f"0b{self.value:0{self.bit_count}b}"
    
    def sign(self):
        return self[-1]
    
    def exponent(self):
        return self[self.f:self.bit_count - 1]
    
    def fraction(self):
        return self[0:self.f]
    
    def to_float(self):
        sign = (-1) ** self.sign().value
        exponent = (self.exponent() - self.bias)[1].value
        fraction = 1 + self.fraction().value / (1 << self.f)
        return sign * fraction * 2 ** exponent





def MUX(data_lines: list[Wire], select_lines: Wire) -> Wire:
    n = len(data_lines)
    output_bit_count = max(data_line.bit_count for data_line in data_lines)
    output = Wire(0, output_bit_count)

    for i in range(n):
        output |= data_lines[i] & (~(select_lines ^ i) * (output_bit_count))

    return output

def SubPos(a: Wire, b: Wire) -> Wire:
    borrow, s = a - b
    return ~borrow, MUX([s, a], borrow)


def IntSqrt(input: Wire) -> Wire:
    if input.bit_count % 2 == 1:
        input = Wire(0,1) // input

    n = input.bit_count
    m = n // 2

    # Initialize the output bits
    q = Wire(0, m)
    pre_layer = input[n-2:n] # last two bits
    for i in range(m-1, -1, -1):
        if i == m - 1:
            q[i], pre_layer = SubPos(pre_layer, Wire(0b01,2))
        else:
            pre_layer = pre_layer // input[2*i : 2*i + 2]
            q[i], pre_layer = SubPos(pre_layer, q[i+1:m] // Wire(0b01,2))

    return q

def FracSqrt(input: Wire) -> Wire:
    return IntSqrt(input // Wire(0, input.bit_count))

def FloatSqrt(input: FloatWire) -> FloatWire:
    S = input.sign()
    E = input.exponent()
    F = input.fraction()
    e = input.e
    f = input.f
    bias = input.bias

    is_zero = (E // F).multi_bit_nor()

    twos_complement_E = (E - bias)[1]

    F_prime = MUX([FracSqrt(MUX([Wire(0b01,2)//F,Wire(1,1)//F//Wire(0,1)],twos_complement_E[0])),Wire(0,f+2)],is_zero)[1:-1]
    E_prime = MUX([(twos_complement_E.SAR()+bias)[1],Wire(0,e)],is_zero)

    return FloatWire((Wire(0,1) // E_prime // F_prime).value, e, f, bias)

def IntMul(a: Wire, b: Wire) -> Wire:
    n = a.bit_count
    m = b.bit_count
    output = Wire(0, n + m)
    for i in range(n):
        for j in range(m):
            output = (output + ((a[i] & b[j]) << (i + j)))[1]
    return output

def FloatMul(a: FloatWire, b: FloatWire) -> FloatWire:
    S = a.sign() ^ b.sign()
    bias = a.bias
    f = a.f
    
    mul = (Wire(1,1) // a.fraction()) * (Wire(1,1) // b.fraction())
    is_F_bigger_than_2 = mul[-1]

    E = (((a.exponent() - bias)[1] + b.exponent())[1] + is_F_bigger_than_2)[1]
    F = MUX([mul[f:-2], mul[f+1:-1]], is_F_bigger_than_2)

    return FloatWire((S // E // F).value, a.e, a.f, a.bias)

def FloatSquare(a: FloatWire) -> FloatWire:
    return FloatMul(a,a)



def mainAlgorithm(a : FloatWire , b : FloatWire) -> FloatWire:
    S_a = a.sign()
    E_a = a.exponent()
    F_a = a.fraction()
    e = a.e # bits of exponent
    f = a.f # bits of fraction
    bias = a.bias
    S_b = b.sign()
    E_b = b.exponent()
    F_b = b.fraction()

    q = Wire(0,1)//bias//Wire(0, 1)# todo : bit//bias//vector

    E_ = b.to_float() - bias.value
    if E_ < 0:
        while E_ < 0:
            a = FloatSqrt(a)
            E_ += 1
            q = FloatMul(a, q) # todo : check this with parsa to see if it is correct (pseudocode wasn't complete)
    elif  E_ > f:
        while E_> f:
            a = FloatSqrt(a)
            E_ -= 1

        F_reverse = F_b.reverse()
        for i in F_reverse.digits : # todo : define a function for traversing digits of a given wire
            a = FloatSqrt(a)
            q = FloatMul(a, q)
            if i == 1:
                q = FloatMul(q, q)

        a = FloatSqrt(a)
        q = FloatMul(q, q)

    else:
        a_prime = Squar_Root_Register # todo : check for more explanation
        F1_prime , F2_prime # todo : check for more explanation

        a_prime = a
        F1_prime_v = Wire(0,1)//Wire(0,1)//Wire(1,0) # todo : check for more information
        F2_prime_v = F_b
        while E_ > 0:
            F1_prime_v << F2_prime_v
            E_ -= 1
            F1_prime = Wire(0,1)
            F2_prime = Wire(0, 1) # todo : check for more info
            for i in range(f,1,-1):
                F1_prime_v >> F1_prime
                F2_prime_v << F2_prime
                a = FloatSquare(a)
                a_prime = FloatSqrt(a_prime)
                if F1_prime == 1:
                    q = FloatMul(q,a)
                if F2_prime == 1:
                    q = FloatMul(q,a_prime)

    return q









# Test cases
a = Wire(0b1011, 4)
b = Wire(0b0101, 4)
print(f'{a = }')
print(f'{a[-4] = }')
print(f'{a[0:3] = }')
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
print(f'{FloatSqrt(a) = } = {FloatSqrt(a).to_float()}\n')  # 1.16

b = FloatWire(0b01000001100000000000000000000000) # 16.0
print(f'{b = } = {b.to_float()}')
print(f'{FloatSqrt(b) = } = {FloatSqrt(b).to_float()}\n')  # 4.0

c = FloatWire(0b00111111100000000000000000000000) # 1.0
print(f'{c = } = {c.to_float()}')
print(f'{FloatSqrt(c) = } = {FloatSqrt(c).to_float()}\n')  # 1.0

d = FloatWire(0b01000000000000000000000000000000) # 2.0
print(f'{d = } = {d.to_float()}')
print(f'{FloatSqrt(d) = } = {FloatSqrt(d).to_float()}\n')  # 1.41

a = Wire(0b1011, 4)
b = Wire(0b0101, 4)
print(f'{IntMul(a, b) = }\n') # 0b00110111

a = FloatWire(0b00111111101011001100110011001101) # 1.35
b = FloatWire(0b01000001100000000000000000000000) # 16.0
print(f'{FloatMul(a, b) = } = {FloatMul(a, b).to_float()}\n') # 0b00111111101011001100110011001101

a = FloatWire(0b00111111101011001100110011001101) # 1.35
print(f'{FloatSquare(a) = } = {FloatSquare(a).to_float()}') # 1.8224999999999998