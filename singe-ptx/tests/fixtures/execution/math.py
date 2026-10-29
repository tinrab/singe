"""Independent rational/Decimal oracles for reciprocal, roots, and exponential.

Rounded reciprocal uses exact fractions; roots use 250 decimal digits.
"""

import json
import random
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

from oracle import FORMATS, decode, encode

HERE = Path(__file__).parent
HEADS = [
    "rcp.approx.f32",
    "rcp.approx.ftz.f32",
    "rcp.approx.ftz.f64",
    "rcp.rm.f32",
    "rcp.rm.f64",
    "rcp.rm.ftz.f32",
    "rcp.rn.f32",
    "rcp.rn.f64",
    "rcp.rn.ftz.f32",
    "rcp.rp.f32",
    "rcp.rp.f64",
    "rcp.rp.ftz.f32",
    "rcp.rz.f32",
    "rcp.rz.f64",
    "rcp.rz.ftz.f32",
    "rsqrt.approx.f32",
    "rsqrt.approx.f64",
    "rsqrt.approx.ftz.f32",
    "rsqrt.approx.ftz.f64",
    "sqrt.approx.f32",
    "sqrt.approx.ftz.f32",
    "sqrt.rm.f32",
    "sqrt.rm.f64",
    "sqrt.rm.ftz.f32",
    "sqrt.rn.f32",
    "sqrt.rn.f64",
    "sqrt.rn.ftz.f32",
    "sqrt.rp.f32",
    "sqrt.rp.f64",
    "sqrt.rp.ftz.f32",
    "sqrt.rz.f32",
    "sqrt.rz.f64",
    "sqrt.rz.ftz.f32",
    "ex2.approx.f32",
    "ex2.approx.ftz.f32",
]
FORMATS["gross"] = (11, 20)


def exponential(head, bits):
    magnitude = bits & 0x7FFFFFFF
    if magnitude > 0x7F800000:
        return 0x7FC00000, "nan", None
    if magnitude == 0x7F800000:
        return (0 if bits >> 31 else magnitude), "exact", None
    if ".ftz." in head and magnitude < 0x00800000:
        return 0x3F800000, "exact", None
    value = decode(bits, "f32")
    if value >= 128:
        return 0x7F800000, "exact", None
    if value <= -150:
        return 0, "exact", None
    with localcontext() as context:
        context.prec = 250
        power = Decimal(value.numerator) / Decimal(value.denominator)
        result = encode(Fraction(Decimal(2) ** power), "f32")
    if ".ftz." in head and result < 0x00800000:
        return 0, "exact", None
    return result, ("exact" if value.denominator == 1 else "ulp2"), None


def expected(head, original):
    op = head.split(".")[0]
    if op == "ex2":
        return exponential(head, original)
    fmt = head.split(".")[-1]
    eb, fb = FORMATS[fmt]
    signbit = 1 << (eb + fb)
    exponent_mask = (1 << eb) - 1
    infinity = exponent_mask << fb
    gross = fmt == "f64" and ".approx.ftz." in head
    bits = original & ~0xFFFFFFFF if gross else original
    sign = bits & signbit
    exponent = (bits >> fb) & exponent_mask
    fraction = bits & ((1 << fb) - 1)
    if exponent == exponent_mask and fraction:
        return 0x7FFFFFFF00000000 if gross else infinity | (1 << (fb - 1)), "nan", None
    if exponent == 0 and (fraction == 0 or ".ftz." in head):
        return (sign if op == "sqrt" else sign | infinity), "exact", None
    if sign and op != "rcp":
        return infinity | (1 << (fb - 1)), "nan", None
    if exponent == exponent_mask:
        return (bits if op == "sqrt" else sign), "exact", None
    value = decode(bits, fmt)
    with localcontext() as context:
        context.prec = 250
        decimal = Decimal(value.numerator) / Decimal(value.denominator)
        if op == "rcp":
            exact = 1 / value
            reference = Decimal(exact.numerator) / Decimal(exact.denominator)
        else:
            reference = decimal.sqrt()
            # Restore exact representable roots before directed rounding.
            # A finite decimal approximation alone can lie just below an exact root.
            candidate = decode(encode(Fraction(reference), fmt), fmt)
            exact = candidate if candidate * candidate == value else Fraction(reference)
            if op == "rsqrt":
                reference = 1 / reference
                exact = 1 / exact
        rnd = head.split(".")[1] if ".approx." not in head else "rn"
        result = encode(exact, "gross" if gross else fmt, rnd) << (32 if gross else 0)
        if ".ftz." in head and result & infinity == 0:
            result &= signbit
        if gross or ".approx." not in head or result & infinity in [0, infinity]:
            return result, "exact", None
        return (
            result,
            {
                "rcp": "ulp1",
                "sqrt": "relative_sqrt",
                "rsqrt": "relative_rsqrt" if fmt == "f32" else "ulp2",
            }[op],
            str(reference),
        )


oracles = []
rng = random.Random(0x805175)
for head in HEADS:
    fmt = head.split(".")[-1]
    eb, fb = FORMATS[fmt]
    width = eb + fb + 1
    four = encode(Fraction(4), fmt)
    infinity = ((1 << eb) - 1) << fb
    one = encode(Fraction(1), fmt)
    two = encode(Fraction(2), fmt)
    values = [
        0,
        1,
        2,
        (1 << fb) - 1,
        1 << fb,
        (1 << fb) + 1,
        one - 1,
        one,
        one + 1,
        two - 1,
        two,
        two + 1,
        four - 1,
        four,
        four + 1,
        infinity - 1,
        infinity,
        infinity + 1,
        infinity | (1 << (fb - 1)),
    ]
    values += [rng.randrange(1, infinity) for _ in range(32)]
    values += [value | (1 << (width - 1)) for value in values[:19]]
    if head.startswith("ex2."):
        # Preserve prior oracle inputs and add dense exponent-boundary checks.
        extra = random.Random(0xE22080)
        for integer in range(-150, 129):
            bits = encode(Fraction(integer), "f32")
            values.extend(
                value for value in (bits - 1, bits, bits + 1) if 0 <= value < 2**32
            )
        values += [
            encode(Fraction(extra.randrange(-150 * 2**20, 128 * 2**20), 2**20), "f32")
            for _ in range(512)
        ]
        values = list(dict.fromkeys(values))
    rows = []
    for bits in values:
        result, comparison, reference = expected(head, bits)
        rows.append(
            dict(
                input=bits, expected=result, comparison=comparison, reference=reference
            )
        )
    oracles.append(dict(head=head, bits=width, cases=rows))

(HERE / "math-oracles.json").write_text(json.dumps(oracles, indent=2) + "\n")
print(
    f"{len(HEADS)} math forms; {sum(len(case['cases']) for case in oracles)} numerical oracle inputs"
)
