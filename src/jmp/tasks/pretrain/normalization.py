"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


class Normalization:
    full = {
        "oc20": {
            "y": {"mean": 1.305542661963295, "std": 24.901469505465872},
            "force": {"mean": 0.0, "std": 0.5111534595489502},
        },
        "oc22": {
            "y": {"mean": 1.232371959806986, "std": 25.229595396538468},
            "force": {"mean": 0.0, "std": 0.25678861141204834},
        },
        "ani1x": {
            "y": {"mean": 0.3835804075000375, "std": 2.8700712783472118},
            "force": {"mean": 0.0, "std": 2.131422996520996},
        },
        "transition1x": {
            "y": {"mean": 0.03843723322120415, "std": 1.787466168382901},
            "force": {"mean": 0.0, "std": 0.3591422140598297},
        },
    }

    linref = {
        "oc20": {
            "y": {"mean": -0.7536948323249817, "std": 2.940723180770874},
            "force": {"mean": 0.0, "std": 0.43649423122406006},
        },
        "oc22": {
            "y": {"mean": 1.2600674629211426, "std": 25.42051887512207},
            "force": {"mean": 0.0, "std": 0.2522418200969696},
        },
        "ani1x": {
            "y": {"mean": 0.3596225082874298, "std": 2.8952934741973877},
            "force": {"mean": 0.0, "std": 2.1361355781555176},
        },
        "transition1x": {
            "y": {"mean": -59.33904266357422, "std": 10.360939979553223},
            "force": {"mean": -2.941862021543784e-06, "std": 0.3591934144496918},
        },
    }

    nolinref = {
        "oc20": {
            "y": {"mean": -359.82421875, "std": 231.93690490722656},
            "force": {"mean": 0.0, "std": 0.43649423122406006},
        },
        "oc22": {
            "y": {"mean": -495.55059814453125, "std": 212.4519805908203},
            "force": {"mean": 0.0, "std": 0.2522418200969696},
        },
        "ani1x": {
            "y": {"mean": -10700.826171875, "std": 3739.096923828125},
            "force": {"mean": 0.0, "std": 2.1361355781555176},
        },
        "transition1x": {
            "y": {"mean": -8254.189453125, "std": 1067.7886962890625},
            "force": {"mean": -2.941862021543784e-06, "std": 0.3591934144496918},
        },
    }
