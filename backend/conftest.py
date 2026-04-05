import sys
import os

#Adding backend/ and services/to path so services is importable
backend_dir = os.path.dirname(__file__)
sys.path.insert(0, backend_dir)
sys.path.insert(0, os.path.join(backend_dir, 'services'))