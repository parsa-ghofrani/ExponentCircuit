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
            return Wire((self.value >> key) & 1, 1)
        elif isinstance(key, slice): # TODO: implement slice for negative values
            start = key.start or 0
            stop = key.stop or (self.bit_count + 1)
            stop -= 1
            mask = (1 << (stop - start)) - 1
            return Wire((self.value >> start) & mask, stop - start)
    
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


def MUX(data_lines: list[Wire], select_lines: Wire) -> Wire:
    n = len(data_lines)
    output_bit_count = max(data_line.bit_count for data_line in data_lines)
    output = Wire(0, output_bit_count)

    for i in range(n):
        output |= data_lines[i] & (~(select_lines ^ i) * (output_bit_count))

    return output

def SubPos(a: Wire, b: Wire) -> Wire:
    b, s = a - b
    return MUX([a, s], ~b)

def IntSqrt(input: Wire) -> Wire:
    pass # TODO: Implement integer square root

# Test cases
a = Wire(0b1011, 4)
b = Wire(0b0101, 4)
print(a)
print(a[-4])
print(a[0:3])
print(a - b)  # 5
print(a + b)  # 15
print(a.concat(b))  # 0b10100101
print(MUX([a, b], Wire(0, 1)))  # 0b1010