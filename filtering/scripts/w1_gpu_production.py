#!/usr/bin/env python
from __future__ import annotations

import sys

from twc.pretraining.w1_gpu_launcher import main


if __name__ == "__main__":
    main(sys.argv[1:])
