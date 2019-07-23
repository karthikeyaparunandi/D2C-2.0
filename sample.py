from mujoco_py import MjSimPool, load_model_from_path

model_xml_string = "/home/karthikeya/Documents/research/model_free_DDP"+"/models/half_cheetah.xml"
self.sim = MjSimPool(load_model_from_path(model_xml_string), device_ids=2)


