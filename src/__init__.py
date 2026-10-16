"""Debris collision avoidance package."""

import sys
import os

package_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(package_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

