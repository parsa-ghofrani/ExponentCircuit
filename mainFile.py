from typing import List
# defining wires
class Wire:
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
            if key < 0 or key >= self.bit_count:
                raise IndexError("Wire index out of range")
            return Wire((self.value >> key) & 1, 1)
        elif isinstance(key, slice): # TODO: implement slice for negative values
            start = key.start or 0
            stop = key.stop or (self.bit_count + 1)
            # stop -= 1
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
    
    def __mul__(self, other: int): # concat n times
        result = self
        for _ in range(other - 1):
            result = result.concat(self)
        return result

    def concat(self, other):
        return Wire((self.value << other.bit_count) | other.value, self.bit_count + other.bit_count)

    def __floordiv__(self, other): # // is concatenation
        return self.concat(other)

    def reverse(self):
        mask = 1 << (self.bit_count - 1)  # Mask to isolate the MSB
        while mask > 0:
            msb = self.value & mask
            self.value = self.value ^ msb
            self.value = self.value << 1
            self.value = self.value | msb
            mask >>= 1
        return self




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

def FloatSqrt(input: Wire) -> Wire:
    pass
    # TODO





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