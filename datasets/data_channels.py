import enum

class LidarChannels(enum.Enum):

    LIDAR_TOP = 'LIDAR_TOP'
    LIDAR_FRONT_LEFT = 'LIDAR_FRONT_LEFT'
    LIDAR_FRONT_RIGHT = 'LIDAR_FRONT_RIGHT'

    @classmethod
    def hasValue(cls, value):
        return value in cls._value2member_map_ 

class ImageChannels(enum.Enum):
    
    CAM_FRONT_ZOOMED = 'CAM_FRONT_ZOOMED'
    CAM_FRONT = 'CAM_FRONT'
    CAM_FRONT_LEFT = 'CAM_FRONT_LEFT'
    CAM_FRONT_RIGHT = 'CAM_FRONT_RIGHT'
    CAM_BACK = 'CAM_BACK'
    CAM_BACK_RIGHT = 'CAM_BACK_RIGHT'
    CAM_BACK_LEFT = 'CAM_BACK_LEFT'

    @classmethod
    def hasValue(cls, value):
        return value in cls._value2member_map_ 