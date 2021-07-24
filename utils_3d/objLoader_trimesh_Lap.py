import trimesh
# from trimesh.scene.scene import Scene
import logging
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)
import numpy as np

class trimesh_load_obj(object):
    def __init__(self, fileName):
        #self.bbox = np.zeros(shape=(2,3))
        ##
        self.vertices = []
        self.faces = []
        self.bbox = []
        # print(fileName)

        obj_info = trimesh.load(fileName, file_type='obj', process=False,use_embree=False, skip_uv=False)

        self.obj_info_Lap = trimesh.smoothing.filter_laplacian(obj_info,iterations = 30)
        # print('hi')
        #self.vertices = obj_info.vertices
        self.vertices = obj_info.vertices #- obj_info.center_mass
        self.vertices = np.array(self.vertices).astype("float32")
        self.faces = np.array(obj_info.faces).astype("int32")
        self.bbox = np.array([[np.max(obj_info.vertices[:,0]), np.max(obj_info.vertices[:,1]), np.max(obj_info.vertices[:,2])], [np.min(obj_info.vertices[:,0]), np.min(obj_info.vertices[:,1]), np.min(obj_info.vertices[:,2])]])

        self.Lap_vertices = self.obj_info_Lap.vertices #- obj_info.center_mass
        self.Lap_vertices = np.array(self.Lap_vertices).astype("float32")
        self.Lap_faces = np.array(obj_info.Lap_faces).astype("int32")
        self.Lap_bbox = np.array([[np.max(obj_info.Lap_vertices[:,0]), np.max(obj_info.Lap_vertices[:,1]), np.max(obj_info.Lap_vertices[:,2])], [np.min(obj_info.Lap_vertices[:,0]), np.min(obj_info.Lap_vertices[:,1]), np.min(obj_info.Lap_vertices[:,2])]])
