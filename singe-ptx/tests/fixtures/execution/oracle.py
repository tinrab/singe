"""Independent exact-rational floating-point encoding and rounding."""

from fractions import Fraction

FORMATS = {"f32": (8, 23), "f64": (11, 52)}


def decode(bits, fmt):
    eb, fb = FORMATS[fmt]
    exponent = (bits >> fb) & ((1 << eb) - 1)
    fraction = bits & ((1 << fb) - 1)
    assert exponent != (1 << eb) - 1
    bias = (1 << (eb - 1)) - 1
    value = Fraction(fraction if exponent == 0 else (1 << fb) + fraction)
    power = (1 - bias if exponent == 0 else exponent - bias) - fb
    value *= Fraction(2) ** power
    return -value if bits >> (eb + fb) else value


def encode(value, fmt, rnd="rn"):
    eb, fb = FORMATS[fmt]
    sign = value < 0
    value = abs(value)
    signbit = int(sign) << (eb + fb)
    if value == 0:
        return signbit
    bias = (1 << (eb - 1)) - 1
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < Fraction(2) ** exponent:
        exponent -= 1
    exponent = max(exponent, 1 - bias)
    scaled = value / Fraction(2) ** (exponent - fb)
    quotient, remainder = divmod(scaled.numerator, scaled.denominator)
    increment = (
        (
            rnd == "rn"
            and (
                2 * remainder > scaled.denominator
                or (2 * remainder == scaled.denominator and quotient & 1)
            )
        )
        or (rnd == "rp" and not sign and remainder)
        or (rnd == "rm" and sign and remainder)
    )
    quotient += bool(increment)
    if quotient >= 1 << (fb + 1):
        exponent += 1
        quotient >>= 1
    if exponent > bias:
        infinity = rnd == "rn" or (rnd == "rp" and not sign) or (rnd == "rm" and sign)
        return signbit | (
            ((1 << eb) - 1) << fb
            if infinity
            else (((1 << eb) - 2) << fb) | ((1 << fb) - 1)
        )
    expfield = exponent + bias if quotient >= (1 << fb) else 0
    return signbit | (expfield << fb) | (quotient & ((1 << fb) - 1))
