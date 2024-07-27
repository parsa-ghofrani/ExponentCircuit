from typing import List
import struct
from collections.abc import Callable
from typing import Union
from copy import deepcopy
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
            stop = key.stop or self.bit_count
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
    
    def twos_complement_value(self):
        return self.value - (1 << self.bit_count) if self.value & (1 << (self.bit_count - 1)) else self.value
    
    def bias_value(self):
        return self.value - 127
    
    def __add__(self, other):
        a = self
        b = other if isinstance(other, Wire) else Wire(other, a.bit_count)
        n = max(a.bit_count, b.bit_count)
        if a.bit_count < b.bit_count:
            a = Wire(0, b.bit_count - a.bit_count) // a
        elif b.bit_count < a.bit_count:
            b = Wire(0, a.bit_count - b.bit_count) // b
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
        result = self[-1] // self
        return result[1:]
    
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
        exponent = (self.exponent() - self.bias)[1].twos_complement_value()
        fraction = 1 + self.fraction().value / (1 << self.f)
        return sign * fraction * 2 ** exponent

registers = []
def do_clock():
    for register in registers:
        register.update_new_value()
        register.set_new_load()
    for register in registers:
        register.update_value()

class Register:
    global registers
    value: Wire | FloatWire
    new_value: Wire | FloatWire
    load: Wire
    data_function: Callable[[], Union[Wire, FloatWire]]
    load_function: Callable[[], Wire]

    def __init__(self, init_value, data_function, load_function) -> None:
        self.value = init_value
        self.new_value = init_value
        self.data_function = data_function
        self.load_function = load_function
        registers.append(self)
        self.load = Wire(0,0)
    
    def update_new_value(self):
        self.new_value = self.data_function()

    def set_new_value(self, new_value):
        self.new_value = new_value

    def set_new_load(self):
        self.load = self.load_function()
    
    def update_value(self):
        if self.load.value == 1:
            self.value = self.new_value





def concat(wires: list[Wire]) -> Wire:
    result = wires[0]
    for wire in wires[1:]:
        result = result // wire
    return result

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

# returns: U, result
# U: underflow
def FloatSqrt(input: FloatWire) -> tuple[Wire, FloatWire]:
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

    result = FloatWire((S // E_prime // F_prime).value, e, f, bias)
    U = F_prime.multi_bit_nor() & Wire(1 if E_prime.value == bias else 0, 1)
    return U, result

def IntMul(a: Wire, b: Wire) -> Wire:
    n = a.bit_count
    m = b.bit_count
    output = Wire(0, n + m)
    for i in range(n):
        for j in range(m):
            output = (output + ((a[i] & b[j]) << (i + j)))[1]
    return output

# returns: O , U , a*b
# O: overflow
# U: underflow
def FloatMul(a: FloatWire, b: FloatWire) -> tuple[Wire, Wire, FloatWire]:
    S = a.sign() ^ b.sign()
    bias = a.bias
    f = a.f
    
    mul = (Wire(1,1) // a.fraction()) * (Wire(1,1) // b.fraction())
    is_F_bigger_than_2 = mul[-1]

    E1 = concat(a.exponent() + b.exponent())
    U, E2 = E1 - bias

    C, E3 = E2 + is_F_bigger_than_2

    O = (E3[-1] | C) & ~U

    # E = (((a.exponent() - bias)[1] + b.exponent())[1] + is_F_bigger_than_2)[1]
    E = E3[:-1]
    F = MUX([mul[f:-2], mul[f+1:-1]], is_F_bigger_than_2)

    return O, U, FloatWire((S // E // F).value, a.e, a.f, a.bias)

# returns: O , U , a^2
# O: overflow
# U: underflow
def FloatSquare(a: FloatWire) -> tuple[Wire, Wire, FloatWire]:
    return FloatMul(a,a)

def SHL(wire: Wire, input_bit: Wire) -> tuple[Wire, Wire]:
    result = wire // input_bit
    return result[-1], result[:-1]

def SHL_two_wires(wire1: Wire, wire2: Wire) -> tuple[Wire, Wire]:
    w2_out_bit, wire2 = SHL(wire2, Wire(0,1))
    _, wire1 = SHL(wire1, w2_out_bit)
    return wire1, wire2

def SHR(wire: Wire, input_bit: Wire) -> tuple[Wire, Wire]:
    result = input_bit // wire
    return result[0], result[1:]


# returns: O , U , a^b
# O: overflow
# U: underflow
def mainAlgorithm(a : FloatWire , b : FloatWire) -> tuple[Wire, Wire, FloatWire]:
    S_a = a.sign()
    E_a = a.exponent()
    F_a = a.fraction()
    e = a.e # bits of exponent
    f = a.f # bits of fraction
    bias = a.bias
    S_b = b.sign()
    E_b = b.exponent()
    F_b = b.fraction()

    O = Wire(0,1)
    U = Wire(0,1)

    real_F = Wire(1,1) // F_b

    q = FloatWire((Wire(0,1)//bias//Wire(0, f)).value)# todo : bit//bias//vector

    E_ = E_b.value - bias.value
    if E_ < 0:
        while E_ < 0:
            u, a = FloatSqrt(a)
            E_ += 1
            if u.value == 1:
                E_ = 0
        for _ in range(f + 1):
            F_output, real_F = SHL(real_F, Wire(0,1))
            u, a = FloatSqrt(a)
            if u.value == 1:
                U = Wire(1,1)
                break
            if F_output.value == 1:
                o, u, q = FloatMul(q, a)
                if o.value == 1:
                    O = Wire(1,1)
                    break
                if u.value == 1:
                    U = Wire(1,1)
                    break
    elif  E_ > f:
        while E_> f:
            o, u, a = FloatSquare(a)
            E_ -= 1
            if o.value == 1 or u.value == 1:
                E = f
        for _ in range(f + 1):
            F_output, real_F = SHR(real_F, Wire(0,1))
            if F_output.value == 1:
                o, u, q = FloatMul(q, a)
                if o.value == 1:
                    O = Wire(1,1)
                    break
                if u.value == 1:
                    U = Wire(1,1)
                    break
            o, u, a = FloatSquare(a)
            if o.value == 1:
                O = Wire(1,1)
                break
            if u.value == 1:
                U = Wire(1,1)
                break

    else:
        a_prime: FloatWire
        F1_prime: Wire
        F2_prime: Wire

        a_prime = deepcopy(a)
        F1_prime = Wire(1, f)
        F2_prime = deepcopy(F_b)
        while E_ > 0:
            # F1_prime_v << F2_prime_v
            # F2_prime_bit_out, F2_prime = SHL(F2_prime, Wire(0,1))
            # _, F1_prime = SHL(F1_prime, F2_prime_bit_out)
            F1_prime, F2_prime = SHL_two_wires(F1_prime, F2_prime)
            E_ -= 1

        for i in range(f,1,-1):
            f1, F1_prime = SHR(F1_prime, Wire(0,1))
            f2, F2_prime = SHL(F2_prime, f1)

            if f1.value == 1:
                o, u, q = FloatMul(q,a)
                if o.value == 1:
                    O = Wire(1,1)
                    break
                if u.value == 1:
                    U = Wire(1,1)
                    break
            o, u, a = FloatSquare(a)
            if o.value == 1:
                O = Wire(1,1)
                break
            if u.value == 1:
                U = Wire(1,1)
                break

            o, u, a_prime = FloatSqrt(a_prime)
            if o.value == 1:
                O = Wire(1,1)
                break
            if u.value == 1:
                U = Wire(1,1)
                break
            if f2.value == 1:
                o, u, q = FloatMul(q,a_prime)
                if o.value == 1:
                    O = Wire(1,1)
                    break
                if u.value == 1:
                    U = Wire(1,1)
                    break

    return O, U, q

def main_algorithm_RTL(a_in : FloatWire , b_in : FloatWire) -> FloatWire:
    e = a_in.e
    f = a_in.f
    bias = a_in.bias

    a = Register(FloatWire(0), None, None)
    S = Register(Wire(0,1), None, None)
    E = Register(Wire(0,e), None, None)
    F = Register(Wire(0,f), None, None)
    a_prime = Register(FloatWire(0), None, None)
    q = Register(FloatWire(0), None, None)
    End = Register(Wire(0,1), None, None)
    F1 = Register(Wire(0,f), None, None)
    F2 = Register(Wire(0,f), None, None)
    i = Register(Wire(0,1), None, None)
    j = Register(Wire(0,1), None, None)
    SC = Register(0, None, None)
    SC_1x = Register(0, None, None)
    RR = Register(FloatWire(0), None, None)
    ShiftCtrl = Register(Wire(0,1), None, None)
    MultiplyCtrl = Register(Wire(0,1), None, None)
    RecCtrl = Register(Wire(0,1), None, None)
    E_l0 = Register(Wire(0,1), None, None) # E < 0
    E_gf = Register(Wire(0,1), None, None) # E > f

    a_LD = Wire(1,1)
    b_LD = Wire(1,1)
    Start = Wire(1,1)

    cycle = 0
    while(End.value == 0):
        if cycle > 0:
            a_LD.value = 0
            b_LD.value = 0
        
        # actual RTL
        if a_LD.value == 1: a.set_new_value(a_in)
        if b_LD.value == 1: b.set_new_value(b_in)
        if a_LD.value == 1: a_prime.set_new_value(a_in)
        if Start.value == 1: 
            End.set_new_value(Wire(0,1))
            q.set_new_value(Wire(0,1)//bias//Wire(0,f))
            E_l0.new_value, E.new_value = E.value - bias
            E_gf.new_value, _ = f - (E.value - bias)[1]
            F1.set_new_value(Wire(1,f))
            F2.set_new_value(F.value)
            i.set_new_value(Wire(0,1))
            j.set_new_value(Wire(0,1))
            ShiftCtrl.set_new_value(Wire(1,1))
            MultiplyCtrl.set_new_value(Wire(0,1))
            RecCtrl.set_new_value(Wire(0,1))
        if ShiftCtrl.value == 1:
            if E_l0.value == 1: a.set_new_value(FloatSqrt(a.value))
            if E_gf.value == 1: a.set_new_value(FloatSquare(a.value))
            if (~E_l0.value & ~E_gf.value).value == 1:
                F1.new_value, F2.new_value = SHL_two_wires(F1.value, F2.value)
            if E_gf.value == 0: E.new_value = E.value - 1
            if E_gf.value == 1: E.new_value = E.value + 1


        if Start.value == 1:
            Start.value = 0
        cycle += 1


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