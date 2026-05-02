import struct

def dec_to_bin(n):
    scaled = int(n * (2**24))
    return f"{scaled:032b}"

def bin_to_dec(b):
    n = int(b, 2)
    return n / (2**24)

class FullAdder:
    def __init__(self, bit_depth=32):
        self.bit_depth = bit_depth

    def _gate_logic(self, a, b, carry_in):
        """Simulates the physical XOR, AND, and OR gates of a Full Adder."""
        sum_bit = (a ^ b) ^ carry_in
        carry_out = (a & b) | (carry_in & (a ^ b))
        return sum_bit, carry_out

    def add(self, bin1, bin2):
        """Performs bitwise addition on two binary strings."""
        result = []
        carry = 0
        
        for i in range(self.bit_depth - 1, -1, -1):
            bit1 = int(bin1[i])
            bit2 = int(bin2[i])
            
            sum_bit, carry = self._gate_logic(bit1, bit2, carry)
            result.append(str(sum_bit))
        
        return "".join(result[::-1])
