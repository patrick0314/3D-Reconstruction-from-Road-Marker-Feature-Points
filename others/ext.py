from Utility import Extrinsic_Camera as EXT
from Utility import ext_cam
import numpy as np

cameras_ext_args = {
    "f": np.array([ 0, 0, 0, -0.50706, 0.47615, -0.48128, 0.53343]),
    "fr": np.array([-0.047226, -0.564235, -0.0429843, -0.13426, 0.75675, -0.63051, 0.10844]),
    "fl": np.array([-0.073756,  0.56121, -0.029907, -0.78108, 0.098631, -0.077138, 0.61175]),
    "b": np.array([-1.5138, -0.043561, -0.016857, 0.48989, 0.51567, -0.50218, -0.49185])
}

cameras_ext = {
    "f": EXT(cameras_ext_args["f"]),
    "fr": EXT(cameras_ext_args["fr"]),
    "fl": EXT(cameras_ext_args["fl"]),
    "b": EXT(cameras_ext_args["b"])
}

for i in cameras_ext:
    print(cameras_ext[i].get_Transform_from_baselink())

