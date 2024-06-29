# defining the gates we need in the project :

# 1 bit not gate
def NOT_GATE(a):
    return ~a + 1


# n bits and gate
def AND_GATE(a, b):
    return a & b


# n bits or gate
def OR_GATE(a, b):
    return a | b


# n bits xor gate
def XOR_GATE(a,b):
    return a ^ b


def NOR_GATE(a, b):
    return NOT_GATE(OR_GATE(a,b))


