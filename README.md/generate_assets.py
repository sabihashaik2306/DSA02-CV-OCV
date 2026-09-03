from pathlib import Path
import numpy as np

out=Path(__file__).resolve().parent/'models'; out.mkdir(exist_ok=True)
# Example intrinsic matrix for development/demo only. Replace with values from the actual camera calibration.
K=np.array([[1000.,0.,640.],[0.,1000.,360.],[0.,0.,1.]])
D=np.array([[-0.20,0.05,0.0,0.0,0.0]])
np.savez(out/'calibration_params.npz',K=K,D=D)
print('Created models/calibration_params.npz (demo parameters; replace with calibrated camera values).')
