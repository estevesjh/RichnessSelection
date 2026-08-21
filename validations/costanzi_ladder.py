"""Attribution ladder: our SigmaPrj total -> full Costanzi-notebook
convention, one flag flip at a time, at (lob=20, zob=0.5).

Rung 0 : our production SigmaPrj (total = rnd + cl), x1e12 to cMpc^-2
Rung 1 : reference port with OUR conventions (wz, slab-xi, DA map,
         cpp NFW, untruncated) -- validates the port machinery
Rung 2 : flip item 7, LoS: wz -> hard +-50 slab
Rung 3 : flip item 6, exclusion: slab-xi -> 3D ball total-zero + floor
Rung 4 : flip item 4, map: DA -> comoving chi
Rung 5 : flip items 1+2, NFW: cpp -> mass-conserving M200m + c(M,z)
Rung 6 : flip item 3, truncation at 30 cMpc/h -> FULL MATTEO

Prints Sigma_total(R) per rung and the step ratio vs the previous rung.
Writes validations/cache/costanzi_ladder.csv.
"""
from __future__ import annotations
import os

import numpy as np

from _common import build_stack, CACHE_DIR
from costanzi_reference import sigma_prj_ref

from richness_selection import SigmaPrj

LOB, ZOB = 20.0, 0.5
R = np.array([0.5, 1.0, 3.0, 10.0, 20.0])

RUNGS = [
    ("1 port, our conventions", dict(los="wz", excl="slab", tmap="DA",
                                     nfw_kind="cpp", trunc=None)),
    ("2 +item7 LoS slab50",     dict(los="slab50", excl="slab", tmap="DA",
                                     nfw_kind="cpp", trunc=None)),
    ("3 +item6 ball exclusion", dict(los="slab50", excl="ball", tmap="DA",
                                     nfw_kind="cpp", trunc=None)),
    ("4 +item4 comoving map",   dict(los="slab50", excl="ball", tmap="chi",
                                     nfw_kind="cpp", trunc=None)),
    ("5 +items1/2 M200m NFW",   dict(los="slab50", excl="ball", tmap="chi",
                                     nfw_kind="m200m", trunc=None)),
    ("6 +item3 trunc 30 = MATTEO", dict(los="slab50", excl="ball",
                                        tmap="chi", nfw_kind="m200m",
                                        trunc=30.0)),
]


def main():
    stack = build_stack()
    cosmo, sb, nfw = stack["cosmo"], stack["sb"], stack["nfw"]

    pre = sb.bias_precompute(LOB, ZOB)
    bsel_fn = sb.marginalised_bias(LOB, ZOB, precomp=pre)

    sp = SigmaPrj(cosmo, sb, nfw)
    dec = sp(R, LOB, ZOB, return_decomposition=True)
    rows = [("0 our SigmaPrj", dec["total"] * 1.0e12, dec["cl"] * 1.0e12)]

    for label, cfg in RUNGS:
        tot, cl = sigma_prj_ref(R, LOB, ZOB, stack, bsel_fn, **cfg)
        rows.append((label, tot, cl))
        print(f"[done] {label}")

    for name, idx in (("TOTAL", 1), ("CL", 2)):
        print(f"\n===== {name} =====")
        print(f"{'rung':34s}" + "".join(f"  R={r:<7.1f}" for r in R))
        prev = None
        for row in rows:
            v = row[idx]
            print(f"{row[0]:34s}" + "".join(f"  {x:.3e}" for x in v))
            if prev is not None:
                print(f"{'   step ratio':34s}"
                      + "".join(f"  {x:7.3f} " for x in v / prev))
            prev = v
    csv = ["rung," + ",".join(f"R={r}" for r in R)]
    for row in rows:
        csv.append(row[0].replace(",", ";") + ",tot,"
                   + ",".join(f"{x:.6e}" for x in row[1]))
        csv.append(row[0].replace(",", ";") + ",cl,"
                   + ",".join(f"{x:.6e}" for x in row[2]))

    out = os.path.join(CACHE_DIR, "costanzi_ladder.csv")
    with open(out, "w") as f:
        f.write("\n".join(csv) + "\n")
    print(f"\n[csv] {out}")


if __name__ == "__main__":
    main()
