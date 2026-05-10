import io
import os
import sys
import argparse
from eval import run

BASE_TRANSLATIONS = os.path.join("data", "manchu", "laoqida.out")
BASE_OUTPUTS = os.path.join("data", "manchu", "outputs")

def run_eval(output_dir, start=0, end=None):
    args = argparse.Namespace(
        output_dir=os.path.join(BASE_OUTPUTS, output_dir),
        ref=BASE_TRANSLATIONS,
        start=start,
        end=end
    )
    buffer = io.StringIO()
    sys.stdout = buffer
    run(args)
    sys.stdout = sys.__stdout__
    return buffer.getvalue()


folders = [f for f in os.listdir(BASE_OUTPUTS) if os.path.isdir(os.path.join(BASE_OUTPUTS, f))]
for folder in folders:
    print(folder)
    result = run_eval(folder)
    lines = result.strip().splitlines()
    print("\n".join(lines[-2:]))