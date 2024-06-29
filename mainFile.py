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


def MUX(n, select_lines, data_lines):
    output = 0

    # Iterating over combinations of select lines
    for i in range(n):
        # Checking if the current select line combination matches the selection and selecting it
        if all(select_lines[j] == ((i >> j) & 1) for j in range(len(select_lines))):
            output = data_lines[i]

    return output


def SUB():
    return
    #TODO