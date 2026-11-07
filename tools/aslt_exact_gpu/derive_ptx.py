"""Derive src/spectrum/aslt_exact_gpu.ptx from the frozen research PTX.

usage: python -I derive_ptx.py <frozen generated.ptx> <out.ptx>

The input is the exact CUDA shader of the 2026-11-03 exact-ASLT research
(integratable_161_20261103/kernel.ptx, SHA-256 below). Kept verbatim: the
module header and the four entries the lane launches (root8, inner8,
whole_phase1_meta, whole_phase2_meta). Every floating-point instruction is
unchanged and checked to be. Changed, integer addressing only:

  * the metadata offset (163968) and the leaf-input offset (243840), which
    the research fixed for 78 outputs a block, become the placeholders
    @@META_OFFSET@@ and @@HEADER_BYTES@@ that the lane fills from its own
    layout constants (one source of truth: aslt_exact_gpu.rs);
  * the hop between requested outputs (267, i.e. 48 kHz at 180 fps), an
    immediate, is read from the metadata word at @@HOP_WORD@@ instead.

The unused legacy entries (whole_phase1, whole_phase2) are not carried, which
also halves what the driver compiles on first use.
"""
import hashlib, re, sys

FROZEN_SHA256 = "8e25fd4302b058b4f595aa220fef70a171978db7badc861905344366a6238c9f"
HEADER = ".version 6.4\n.target sm_75\n.address_size 64\n"
KEEP = ["root8", "inner8", "whole_phase1_meta", "whole_phase2_meta"]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def once(s, a, b):
    assert s.count(a) == 1, (a[:100], s.count(a))
    return s.replace(a, b, 1)


def entries(text):
    starts = [(m.group(1), m.start()) for m in re.finditer(r"\.visible \.entry\s+(\w+)\(", text)]
    out = {}
    for (name, a), (_, b) in zip(starts, starts[1:] + [("", len(text))]):
        out[name] = text[a:b]
    return out


def fp(x):
    return "\n".join(re.findall(r"^\s*(?:add|sub|mul|fma|neg|abs)\.[a-z.]*f32[^\n]*|(?:add|sub|mul|fma)\.rn\.f32[^;]*;", x, re.M))


def main():
    src, dst = sys.argv[1], sys.argv[2]
    data = open(src, "rb").read()
    assert sha(data) == FROZEN_SHA256, "not the frozen research PTX"
    text = data.decode("utf-8")
    assert text.count(HEADER) == 1 and ".ftz" not in text
    e = entries(text)
    assert list(e) == ["root8", "inner8", "whole_phase1", "whole_phase2", "whole_phase1_meta", "whole_phase2_meta"], list(e)
    kept = {n: e[n] for n in KEEP}
    p1 = kept["whole_phase1_meta"]
    p1 = once(p1, ".reg .b32 %meta_first,%meta_m,%meta_j,%meta_p,%meta_o,%meta_r0,%meta_r1,%meta_u,%meta_y;",
                  ".reg .b32 %meta_first,%meta_m,%meta_j,%meta_p,%meta_o,%meta_r0,%meta_r1,%meta_u,%meta_y,%meta_hop;")
    p1 = once(p1, "add.u64 %descriptor,%input,163968;ld.global.u32 %meta_first,[%descriptor];ld.global.u32 %meta_m,[%descriptor+4];",
                  "add.u64 %descriptor,%input,@@META_OFFSET@@;ld.global.u32 %meta_first,[%descriptor];ld.global.u32 %meta_m,[%descriptor+4];"
                  "ld.global.u32 %meta_hop,[%descriptor+@@HOP_WORD@@];")
    p1 = once(p1, "mad.lo.u32 %meta_o,%meta_j,267,%meta_first;", "mad.lo.u32 %meta_o,%meta_j,%meta_hop,%meta_first;")
    p1 = once(p1, "add.u64 %packet_input,%packet_input,243840;", "add.u64 %packet_input,%packet_input,@@HEADER_BYTES@@;")
    p2 = kept["whole_phase2_meta"]
    p2 = once(p2, "add.u64 %descriptor,%input,163968;", "add.u64 %descriptor,%input,@@META_OFFSET@@;")
    for name, new in (("whole_phase1_meta", p1), ("whole_phase2_meta", p2)):
        assert fp(new) == fp(kept[name]), f"{name}: a floating-point instruction changed"
        for c in ("163968", "243840"):
            assert c not in new, (name, c)
        kept[name] = new
    assert re.search(r"\b267\b", "".join(kept.values())) is None
    body = "".join(kept[n] for n in KEEP)
    out = ("// Moosik exact ASLT GPU lane. Derived by tools/aslt_exact_gpu/derive_ptx.py from the\n"
           f"// frozen research PTX {FROZEN_SHA256};\n"
           "// floating-point instructions unchanged, integer addressing parameterised (see the script).\n"
           + HEADER + body)
    if not out.endswith("\n"):
        out += "\n"
    open(dst, "w", encoding="utf-8", newline="\n").write(out)
    print({"out": dst, "bytes": len(out.encode()), "sha256": sha(out.encode()),
           "fp_lines_sha256": {n: sha(fp(kept[n]).encode()) for n in KEEP}})


if __name__ == "__main__":
    main()
