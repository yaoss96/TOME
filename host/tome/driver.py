"""Driving the ToME kernel on the ZCU104 from PYNQ.

The dense operand S is stored as FP8 E4M3 with one global shared exponent
sigma: the board holds q = E4M3(v * 2^-sigma) with
sigma = floor(log2 max|S|) - floor(log2 0.9375), and the kernel folds
2^sigma into each frontier value (register b_sexp).  A 128-bit beat then
carries 16 operands.

The frontier travels as MP-COO packets (mpcoo.py): the host packs only the
initial frontier; every later hop's input is the kernel's own output.
"""
import math
import time

import numpy as np

VEC_W = 16                      # FP8 operands per 128-bit beat
E4M3_MAX = 0.9375
E4M3_FEXP = -1                  # floor(log2(0.9375))

# Buffer capacities of the synthesized kernel (hls/tome/spmm_accel.h)
MAX_TOP_K = 200
MAX_ROW_SPAN = 256
MAX_RES_PKTS = MAX_ROW_SPAN * MAX_TOP_K
RES_BUF_SIZE = MAX_RES_PKTS + 1   # packets; word 0 of packet 0 holds the count
MAX_A_PKT = 16384                 # input packets per launch

# AXI-lite register map of the 4-PE kernel (xspmm_kernel_hw.h from HLS)
REGS = dict(ctrl=0x00, a=0x10, bp=[0x1c, 0x28, 0x34, 0x40], res=0x4c,
            tpkts=0x58, rbase=0x60, M=0x68, K=0x70, N=0x78, rspan=0x80,
            tn=0x88, topk=0x90, lock=0x98, bsexp=0xa0)

SENTINEL_W = 0xFFFFFFFF   # top two bits 0b11 = PREC_SENTINEL, never emitted


def enc_e4m3(v):
    """float -> E4M3 byte (sign always 0 here; S is non-negative)"""
    if v <= 0.0:
        return 0
    eb, mb, bias, exp_max, mant_max = 4, 3, 16, 15, 7
    fr, e = math.frexp(v)
    e -= 1
    fr *= 2.0
    ebv = e + bias
    if ebv > exp_max:
        return (exp_max << mb) | mant_max
    if ebv <= 0:
        mf = v / (2.0 ** (1 - bias)) * (1 << mb)
        return min(int(mf + 0.5), mant_max)
    m = int((fr - 1.0) * (1 << mb) + 0.5)
    if m > mant_max:
        m = 0
        ebv += 1
        if ebv > exp_max:
            return (exp_max << mb) | mant_max
    return (ebv << mb) | m


def enc_e4m3_vec(a):
    """vectorised E4M3 encode; must agree with enc_e4m3 element-wise"""
    out = np.zeros(a.shape, dtype=np.uint8)
    nz = a > 0
    if not nz.any():
        return out
    v = a[nz].astype(np.float64)
    e = np.floor(np.log2(v)).astype(np.int64)
    ebv = e + 16
    # Branch on the PRE-carry exponent, exactly as mfp_encode does: a value
    # just below 2^-15 takes the subnormal path and clamps its mantissa,
    # rather than being rounded up into the normal range.
    sub = ebv <= 0
    norm = ~sub

    code = np.zeros(v.shape, dtype=np.int64)
    if norm.any():
        vn, en, ebn = v[norm], e[norm], ebv[norm]
        frac = vn / np.exp2(en.astype(np.float64))      # in [1,2)
        m = np.rint((frac - 1.0) * 8.0).astype(np.int64)
        carry = m > 7                                   # rounded up to 2.0
        m[carry] = 0
        ebn = ebn + carry.astype(np.int64)
        cn = (np.clip(ebn, 1, 15) << 3) | np.clip(m, 0, 7)
        cn[ebn > 15] = (15 << 3) | 7                    # saturate
        code[norm] = cn
    if sub.any():
        mf = v[sub] / (2.0 ** -15) * 8.0
        code[sub] = np.clip(np.floor(mf + 0.5), 0, 7).astype(np.int64)
    out[nz] = code.astype(np.uint8)
    return out


def encode_s(S):
    """(sigma, N x Nhw uint8 E4M3 image of S, rows padded to VEC_W)"""
    n = S.shape[0]
    nhw = ((n + VEC_W - 1) // VEC_W) * VEC_W
    sigma = int(math.floor(math.log2(float(np.abs(S).max())))) - E4M3_FEXP
    Bp = np.zeros((n, nhw), dtype=np.uint8)
    # Encode in row blocks: enc_e4m3_vec needs several float64 temporaries
    # the size of its input, which a 2-GB board cannot afford for all of S.
    blk = max(1, int(4e6 // max(n, 1)))
    for r0 in range(0, n, blk):
        r1 = min(r0 + blk, n)
        Bp[r0:r1, :n] = enc_e4m3_vec(np.asarray(S[r0:r1], np.float64) * 2.0 ** -sigma)
    return sigma, Bp


def pkts2buf(buf, pkts, off=0):
    """128-bit packets -> four little-endian 32-bit words each"""
    for i, p in enumerate(pkts):
        b = (off + i) * 4
        buf[b]     = int(p)       & 0xFFFFFFFF
        buf[b + 1] = int(p >> 32) & 0xFFFFFFFF
        buf[b + 2] = int(p >> 64) & 0xFFFFFFFF
        buf[b + 3] = int(p >> 96) & 0xFFFFFFFF


def buf2pkt(buf, idx):
    b = idx * 4
    return (int(buf[b]) |
            (int(buf[b + 1]) << 32) |
            (int(buf[b + 2]) << 64) |
            (int(buf[b + 3]) << 96))


def extract_pkts(rb, max_pkts=MAX_RES_PKTS):
    """Output packets of one launch.

    The kernel writes its packets from index 1 and then the count + 1 to
    word 0.  The host fills the buffer with SENTINEL_W first, so if the
    count is ever implausible the packets can still be found by scanning
    for the sentinel tier."""
    n_hdr = int(rb[0]) - 1
    if 0 < n_hdr <= max_pkts:
        return [buf2pkt(rb, 1 + i) for i in range(n_hdr)], n_hdr, 'header'
    out = []
    for i in range(max_pkts):
        w3 = int(rb[(1 + i) * 4 + 3])
        if w3 == SENTINEL_W or (w3 >> 30) == 0x3:
            break
        out.append(buf2pkt(rb, 1 + i))
    return out, len(out), 'scan'


class Kernel:
    """The spmm_kernel IP: program its registers, start it, wait for done."""

    def __init__(self, mmio, regs=REGS):
        self.m = mmio
        self.r = regs

    @staticmethod
    def from_overlay(overlay, ip='spmm_kernel_0'):
        return Kernel(getattr(overlay, ip).mmio)

    def _w(self, o, v):
        self.m.write(o, int(v) & 0xFFFFFFFF)

    def call(self, ab, bbs, rb, npkt, rbase, M, K, Nhw, rs, tn, tk, lk, bsexp):
        """One launch over one row panel; returns ap_start-to-ap_done seconds.

        ab: input packets, rb: output packets, bbs: one buffer per b_port.
        Every port must be given S's address: all four PEs read S, through
        their own ports, from the same buffer behind one shared HP port."""
        if not isinstance(bbs, (list, tuple)):
            bbs = [bbs] * len(self.r['bp'])
        ab.flush(); rb.flush()
        seen = set()
        for b in bbs:
            if id(b) not in seen:
                b.flush(); seen.add(id(b))
        pa, pr = int(ab.physical_address), int(rb.physical_address)
        r = self.r
        t0 = time.time()
        self._w(r['a'], pa & 0xFFFFFFFF); self._w(r['a'] + 4, pa >> 32)
        for off, b in zip(r['bp'], bbs):
            pb = int(b.physical_address)
            self._w(off, pb & 0xFFFFFFFF); self._w(off + 4, pb >> 32)
        self._w(r['res'], pr & 0xFFFFFFFF); self._w(r['res'] + 4, pr >> 32)
        for key, val in (('tpkts', npkt), ('rbase', rbase), ('M', M), ('K', K),
                         ('N', Nhw), ('rspan', rs), ('tn', tn), ('topk', tk),
                         ('lock', lk), ('bsexp', bsexp)):
            self._w(r[key], val)
        self._w(r['ctrl'], 0x01)
        while not (self.m.read(r['ctrl']) & 0x02):
            pass
        t1 = time.time()
        rb.invalidate()
        return t1 - t0


# ------------------------------------------------------------ PL clock
CLKACT = 1 << 24          # CRL_APB PL0_REF_CTRL: clock active


def pl0_mhz():
    """PL0 as the hardware has it: CRL_APB PL0_REF_CTRL over its PLL.

    A bitstream's PS configuration may derive PL0 from a PLL other than
    IOPLL; PYNQ applies the divisors to IOPLL regardless, so the frequency
    asked for is not always the one the fabric gets.  Always read it back."""
    from pynq import MMIO
    crl, crf = MMIO(0xFF5E0000, 0x200), MMIO(0xFD1A0000, 0x100)

    def pll(v):
        return 100.0 / 3.0 * ((v >> 8) & 0x7F) / (2 if (v >> 16) & 1 else 1)
    c = crl.read(0xC0)
    src = c & 7
    f = {0: pll(crl.read(0x20)), 2: pll(crl.read(0x30))}.get(src)
    if src == 3:
        f = pll(crf.read(0x2C)) / ((crf.read(0x68) >> 8) & 0x3F)
    return f / ((c >> 8) & 0x3F) / ((c >> 16) & 0x3F)


def gate(on):
    """Start or stop PL0.  Stopped, the fabric keeps its configuration but
    nothing in it toggles.  Nothing may touch the PL while it is stopped."""
    from pynq import MMIO
    crl = MMIO(0xFF5E0000, 0x200)
    c = crl.read(0xC0)
    crl.write(0xC0, (c | CLKACT) if on else (c & ~CLKACT))
    return bool(crl.read(0xC0) & CLKACT)


def set_mhz(mhz):
    """Set PL0 and verify it from the clock registers."""
    from pynq import Clocks
    gate(True)
    Clocks.fclk0_mhz = mhz
    gate(True)
    got = pl0_mhz()
    if abs(got - mhz) > 0.5:
        raise SystemExit(f'PL0 is {got:.2f} MHz, asked for {mhz}')
    return got
