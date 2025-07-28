from mujoco import MjModel, MjData
from mujoco.viewer import launch_passive
import mujoco

# 載入 MJCF 模型
model = MjModel.from_xml_path("brown_cylinder.xml")
data = MjData(model)

# 開啟 GUI viewer
with launch_passive(model, data) as viewer:
    for _ in range(1000000000):
        mujoco.mj_step(model, data)
    
