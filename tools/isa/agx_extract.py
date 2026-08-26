#!/usr/bin/env python3
"""Pull the native AGX program out of a GPU binary produced by ``metal-tt``.

``metal-tt`` writes a fat Mach-O with one GPU slice per architecture.  Each
slice carries a ``__compute`` section whose payload is itself a Mach-O GPU
executable holding the shader's ``__text``.  This walks both levels without
needing a Metal device.

    python3 tools/isa/agx_extract.py --arch applegpu_g15s in.gpubin -o out/
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import sys

FAT_MAGIC_METAL = 0xCBFEBABE
FAT_CIGAM_METAL = 0xBEBAFECB
MH_MAGIC_64 = 0xFEEDFACF
MH_CIGAM_64 = 0xCFFAEDFE
LC_SEGMENT_64 = 0x19
LC_SYMTAB = 0x02

# cpusubtype -> Apple GPU architecture name, learned from `metal-lipo -info`
# on a fat binary built for one family at a time.
ARCH_BY_SUBTYPE: dict[int, str] = {}


def _u32(b: bytes, off: int) -> int:
    return struct.unpack_from("<I", b, off)[0]


def slices(data: bytes) -> list[tuple[int, int, int, int]]:
    """Return (offset, size, cputype, cpusubtype) for each Mach-O in ``data``."""

    magic = struct.unpack_from(">I", data, 0)[0]
    if magic in (FAT_MAGIC_METAL, FAT_CIGAM_METAL):
        # Metal's fat header is little-endian despite the classic magic.
        nfat = _u32(data, 4)
        out = []
        for i in range(nfat):
            base = 8 + i * 20
            cputype, cpusub, off, size, _align = struct.unpack_from(
                "<iiIII", data, base
            )
            out.append((off, size, cputype, cpusub))
        return out
    return [(0, len(data), _u32(data, 4), _u32(data, 8))]


def sections(macho: bytes) -> dict[str, tuple[int, int]]:
    """Map ``segment,section`` -> (file offset, size) inside one Mach-O."""

    magic = _u32(macho, 0)
    if magic not in (MH_MAGIC_64, MH_CIGAM_64):
        raise ValueError(f"not a 64-bit Mach-O (magic {magic:#x})")
    ncmds = _u32(macho, 16)
    off = 32
    out: dict[str, tuple[int, int]] = {}
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", macho, off)
        if cmd == LC_SEGMENT_64:
            nsects = _u32(macho, off + 64)
            soff = off + 72
            for _s in range(nsects):
                sectname = macho[soff : soff + 16].rstrip(b"\0").decode()
                segname = macho[soff + 16 : soff + 32].rstrip(b"\0").decode()
                _addr, size = struct.unpack_from("<QQ", macho, soff + 32)
                fileoff = _u32(macho, soff + 48)
                out[f"{segname},{sectname}"] = (fileoff, size)
                soff += 80
        off += cmdsize
    return out


def symbols(macho: bytes) -> list[tuple[str, int, int]]:
    """Return (name, section index, value) from the symbol table."""

    ncmds = _u32(macho, 16)
    off = 32
    out: list[tuple[str, int, int]] = []
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", macho, off)
        if cmd == LC_SYMTAB:
            symoff, nsyms, stroff, strsize = struct.unpack_from("<IIII", macho, off + 8)
            strtab = macho[stroff : stroff + strsize]
            for i in range(nsyms):
                base = symoff + i * 16
                n_strx, n_type, n_sect = struct.unpack_from("<IBB", macho, base)
                value = struct.unpack_from("<Q", macho, base + 8)[0]
                end = strtab.find(b"\0", n_strx)
                name = strtab[n_strx:end].decode(errors="replace")
                if name:
                    out.append((name, n_sect, value))
        off += cmdsize
    return out


def extract(path: pathlib.Path, arch_index: int | None, want_subtype: int | None):
    data = path.read_bytes()
    found = []
    for off, size, cputype, cpusub in slices(data):
        if want_subtype is not None and cpusub != want_subtype:
            continue
        macho = data[off : off + size]
        try:
            sects = sections(macho)
        except ValueError:
            continue
        comp = sects.get("__TEXT,__compute")
        if comp is None:
            continue
        inner = macho[comp[0] : comp[0] + comp[1]]
        found.append((cputype, cpusub, inner, sects, macho))
    if arch_index is not None:
        found = [found[arch_index]]
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--outdir", default=None)
    ap.add_argument("--subtype", type=lambda s: int(s, 0), default=None)
    ap.add_argument("--index", type=int, default=None)
    args = ap.parse_args()

    path = pathlib.Path(args.input)
    found = extract(path, args.index, args.subtype)
    if not found:
        print("no __compute section found", file=sys.stderr)
        return 1

    outdir = pathlib.Path(args.outdir) if args.outdir else path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    for cputype, cpusub, inner, sects, _macho in found:
        tag = f"{cputype:#x}_{cpusub:#x}"
        isects = sections(inner)
        text = isects.get("__TEXT,__text")
        syms = symbols(inner)
        if text is None:
            print(f"{tag}: nested Mach-O has no __text", file=sys.stderr)
            continue
        code = inner[text[0] : text[0] + text[1]]
        out = outdir / f"{path.stem}.{tag}.text.bin"
        out.write_bytes(code)
        names = ", ".join(f"{n}@{v:#x}" for n, _s, v in syms) or "-"
        print(f"{out}  text={len(code)}B  symbols: {names}")
        for key in ("__TEXT,__descriptor", "__TEXT,__reflection"):
            if key in sects:
                o, s = sects[key]
                (outdir / f"{path.stem}.{tag}.{key.split(',')[1]}.bin").write_bytes(
                    _macho_slice(_macho, o, s)
                )
    return 0


def _macho_slice(macho: bytes, off: int, size: int) -> bytes:
    return macho[off : off + size]


if __name__ == "__main__":
    raise SystemExit(main())
