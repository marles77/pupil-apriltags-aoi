# ======================================================================= #
# analyze gaze data from Pupil Core eye tracker using apriltag detection  #
#                                                                         #
# ========================================================================#

import yaml
import sys
from collections import defaultdict
from pupil_apriltags import Detector
import cv2
import numpy as np
import pandas as pd
from pprint import pprint
import logging
from typing import List, Optional, Any, Dict, Tuple, Union
#from pydantic import BaseModel
from dataclasses import dataclass, field
from datetime import datetime


class SingletonMeta(type):
    """
    A metaclass for creating a Singleton base class.
    """
    _instances: dict = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class Cols:
    CRED = '\033[91m'
    CGREEN  = '\33[32m'
    CYELLOW = '\33[33m'
    CEND = '\033[0m'

class Constants:
    # april tags
    TAGS = {
            'FAMILY': "tag36h11",
            'IDS': (20, 17, 22, 19, 23, 18),
            'CORNERS': {20: 0, 17: 3, 22: 0.5, 19: 1.5, 23: 1, 18: 0},
            }
    Y_CORRECT = 0.25

    COORD = {
            0.5: (0, 1),
            1.5: (0, 3)
            }

    # paths to video files and gaze data files
    PATH_VID = "video/"
    FILE = "world.mp4"
    VID = PATH_VID + FILE # 0 for webcam
    PATH_GAZE = "gaze/"
    FILE_GAZE = "gaze_positions.csv"
    GAZE = PATH_GAZE + FILE_GAZE
    TABLE_GAZE = "gaze_table"
    FRAME_START = 0
    FRAME_STOP = None


@dataclass
class AppSettings:
    # setting1: str = field(default="default_value1")
    # setting2: int = field(default=42)
    # setting3: bool = field(default=True)
    # tags
    TAGS: defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))
    X_CORRECT: float = field(default=0)
    Y_CORRECT: float = field(default=0.25)
    COORD: defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))

    # paths to video files and gaze data files
    PATH_VID: str = field(default='')
    FILE: str = field(default='')
    VID: Any = field(default='')
    PATH_GAZE: str = field(default='')
    FILE_GAZE: str = field(default='')
    GAZE: str = field(default='')
    TABLE_GAZE: str = field(default='')
    FRAME_START: int = field(default=0)
    FRAME_STOP: Optional[int|None] = field(default=None)



class Settings(AppSettings, metaclass=SingletonMeta):
    """
    The AppSettings class with Singleton behavior.
    """
    @classmethod
    def load_from_yaml(cls, yaml_file: str):
        # Ensure instance is created before attempting to load from YAML
        if cls not in SingletonMeta._instances:
            SingletonMeta._instances[cls] = cls()

        with open(yaml_file, 'r') as file:
            config_data = yaml.safe_load(file)
        
        instance = SingletonMeta._instances[cls]
        
        for key, value in config_data.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
    

    def update_dirs(self):
        if self.FILE:
            self.VID = self.PATH_VID + self.FILE
        else:
            print(f"{Cols.CRED}Brak pliku video: *{self.FILE}* {Cols.CEND}")
            sys.exit()
        if self.FILE_GAZE:
            self.GAZE = self.PATH_GAZE + self.FILE_GAZE
        else:
            print(f"{Cols.CRED}Brak pliku gaze{Cols.CEND}")
            sys.exit()
        


now = datetime.now()
datetime_str = now.strftime("%Y%m%d_%H%M%S")

# Configure logging
logging.basicConfig(filename='log/april_error.log',
                    level=logging.ERROR,
                    format='%(asctime)s %(levelname)s %(message)s')

# configure april tag detector
detector = Detector(
   families=Constants.TAGS['FAMILY'],
   nthreads=1,
   quad_decimate=1.0,
   quad_sigma=0.0,
   refine_edges=1,
   decode_sharpening=0.25,
   debug=0
)


@dataclass
class Aoi:
    '''
    Area of interest class
    '''
    name: str
    tag_ids: Optional[List[int]|None] = None
    #pts: Optional[List[List[float]]]
    vertices = Optional[List[List[float|int]]|None]
    gaze_sum: int = 0

    def update(self, tags_selected: Dict, point: Tuple[int, int]) -> None:
        if (len( tags_to_aoi := {tag.tag_id: tag for _, tag in tags_selected.items() if tag.tag_id in self.tag_ids} ) in (3, 4)):
            #print(f"AOI: {self.name}; tags: {tags_to_aoi.keys()}")
            pts = self._establish_pts(tags=tags_to_aoi, ids=self.tag_ids)
            if [None, None] in pts:
                pts.append(self._find_fourth_vertex(pts))
                pts.remove([None, None])
        
            self.vertices = self._sort_vertices(pts)
            if Gaze.is_point_in_quadrilateral(point[0], point[1], self.vertices):
                self.gaze_sum += 1
                #print(f"{CRED}Left AOI{CEND}")
                return 1
            else:
                return 0
                #print(f"{CYELLOW}No AOI{CEND}")
        return 0

    def _find_fourth_vertex(self, pts: List[List[float]]) -> Optional[List[float]]:
        '''
        Finds the fourth vertex of the quadrilateral (aoi)
        parameters:
            pts: list of points of the quadrilateral
        returns:
            pt: [list] fourth vertex of the quadrilateral
        '''
        pt =[None, None]
        try: 
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pts

            if x4 is None:
                x4, y4 = x1 + x3 - x2, y1 + y3 - y2
                pt = [x4, y4]
            elif x3 is None:
                x3, y3 = -x1 + x2 + x4, -y1 + y2 + y4
                pt = [x3, y3]   
            elif x2 is None:
                x2, y2 = x1 - x4 + x3, y1 - y4 + y3
                pt = [x2, y2]   
            elif x1 is None:
                x1, y1 = x2 - x3 + x4, y2 - y3 + y4
                pt = [x1, y1]
            else:
                raise Exception("Error finding fourth vertex")
        except Exception as e:
            logging.error(f"Error finding fourth vertex: {e}")
            # print(f"Error finding fourth vertex: {e}")
            # exit()
        return pt
    
    def _sort_vertices(self, vertices: List[List[float]]) -> List[List[float]]:
        '''
        Sorts vertices based on the angle with the centroid
        parameters:
        vertices: list of vertices of the quadrilateral (aoi)
        '''
        try:
            # Calculate the centroid of the polygon
            centroid = tuple(map(lambda x: sum(x) / len(vertices), zip(*vertices)))
            # Sort vertices based on the angle with the centroid
            vertices.sort(key=lambda vertex: (np.arctan2(vertex[1] - centroid[1], vertex[0] - centroid[0])))
        except Exception as e:
            logging.error(f"Error sorting vertices: {e}")
            return vertices
            # print(f"Error sorting vertices: {e}")
            # exit()
        return vertices
    
    def _establish_pts(self, tags: Dict, ids: Tuple) -> List[List[float]]:
        '''
        Establishes the vertices of the quadrilateral (aoi)
        parameters:
        tags: dictionary of apriltag ids and their corresponding apriltag objects
        '''
        pts = []
        tags_sorted = [tags.get(id, None) for id in ids]

        try:
            
            for tag in tags_sorted:
                if tag == None:
                    pt = (None, None)
                else:
                    size = abs(tag.corners[0][1] - tag.corners[2][1])
                    corner = Constants.TAGS['CORNERS'][tag.tag_id]
                    if corner in (0, 1, 2, 3):
                        pt = list(tag.corners[Constants.TAGS['CORNERS'][tag.tag_id]].astype(int))              
                        pt[1] = pt[1] + int(size * Constants.Y_CORRECT) if tag.tag_id in (20, 22, 23) else pt[1] - int(size * Constants.Y_CORRECT)
                    else:
                        pt = list((tag.center[0].astype(int), tag.corners[Constants.COORD[corner][0]][1].astype(int)))
                        pt[1] = pt[1] + int(size * Constants.Y_CORRECT) if tag.tag_id in (20, 22, 23) else pt[1] - int(size * Constants.Y_CORRECT)

                pts.append(pt)
        except Exception as e:
            logging.error(f"Error establishing pts: {e}")
            return pts
            # print(f"Error establishing pts: {e}")
            # exit()
        return pts
    
    def _transform_vertices_to_opencv(self, vertices: List[Tuple[float, float]], frame_width: int, frame_height: int) -> List[Tuple[int, int]]:
        """
        Transform the vertices of the quadrilateral to the OpenCV coordinate system.

        :param vertices: List of tuples [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] in the original coordinate system
        :param frame_width: Width of the OpenCV frame
        :param frame_height: Height of the OpenCV frame
        :return: List of tuples in the OpenCV coordinate system
        """
        return [Gaze.transform_point_to_opencv(vertex, frame_width, frame_height) for vertex in vertices]



@dataclass
class Gaze():
    '''
    Class to store gaze data
    '''
    path: str
    data: np.ndarray = None
    compr_data: np.ndarray = None # Compressed gaze data - to match world camera frames
    data_table: pd.DataFrame = None # Pandas table to store the gaze data and aoi data
    
    def __post_init__(self):
        try:
            self.data = np.genfromtxt(Constants.GAZE, delimiter=",", dtype=float, skip_header=1, skip_footer=1)
            index_column = 1
            # Get unique indices and their first occurrence positions
            _, unique_pos = np.unique(self.data[:, index_column], return_index=True)
            # Select the rows corresponding to the first occurrence of each unique index
            self.compr_data = self.data[unique_pos]
            # Pandas table
            self.data_table = pd.read_csv(Constants.GAZE, sep=",")
            self.data_table['aoi-left'] = [None] * self.data_table.shape[0]
            self.data_table['aoi-right'] = [None] * self.data_table.shape[0]
            self.data_table['aoi-gaze'] = [''] * self.data_table.shape[0]
            ###
            print(f"{Cols.CYELLOW}Compressed Rows: {self.compr_data.shape[0]} | 1st: {self.compr_data[0, 1]} | Last: {self.compr_data[-1, 1]} | df: {self.data_table.shape}{Cols.CEND}")
            print(self.data_table.head())
            print(self.data_table.info())
            ###
        except Exception as e:
            #logging.error(f"Error reading gaze data: {e}")
            print(f"Error reading gaze data: {e}")
            exit()
        finally:
            print(f"{Cols.CGREEN}Gaze data loaded | Rows: {self.data.shape[0]}{Cols.CEND}")
    

    def update_table(self, frame: int, aoi_data: Dict, aoi_gaze: str) -> None:
        pass


    @staticmethod
    def transform_point_to_opencv(point: Tuple[float, float], frame_width: int, frame_height: int) -> Tuple[int, int]:
        """
        Transform a point from the original coordinate system to the OpenCV coordinate system.

        :param point: Tuple (x, y) in the original coordinate system (0,0 to 1,1)
        :param frame_width: Width of the OpenCV frame
        :param frame_height: Height of the OpenCV frame
        :return: Tuple (new_x, new_y) in the OpenCV coordinate system
        """
        point_x, point_y = point
        new_x = int(point_x * frame_width)
        new_y = int((1 - point_y) * frame_height)
        return new_x, new_y
    
    @staticmethod
    def is_point_in_quadrilateral(px: int, py: int, vertices: List[Tuple[int, int]]) -> bool:
        """
        Determine if the point (px, py) is inside the quadrilateral defined by vertices.

        :param px: x coordinate of the point
        :param py: y coordinate of the point
        :param vertices: List of tuples [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        :return: True if the point is inside the quadrilateral, False otherwise
        """

        try:
            n = len(vertices)
            inside = False

            xints = 0.0
            p1x, p1y = vertices[0]
            for i in range(n + 1):
                p2x, p2y = vertices[i % n]
                if py > min(p1y, p2y):
                    if py <= max(p1y, p2y):
                        if px <= max(p1x, p2x):
                            if p1y != p2y:
                                xints = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or px <= xints:
                                inside = not inside
                p1x, p1y = p2x, p2y

            return inside
        except Exception as e:
            logging.error(f"Error determining if point is inside quadrilateral: {e}")
            # print(f"Error determining if point is inside quadrilateral: {e}; frame: {num_frame}")
            # exit()
            return False


def run(vid=Constants.VID, detector=detector, tag_ids=Constants.TAGS['IDS']) -> None:
    '''
    Main function to run apriltag detection
    parameters:
    vid: video file path; if 0, webcam will be used
    '''
    gaze = Gaze(path=Constants.GAZE)
    left_aoi = Aoi(name="left", tag_ids=tag_ids[:4])
    right_aoi = Aoi(name="right", tag_ids=tag_ids[2:6])
    white_space_aoi = Aoi(name="white-space")
    #exit()
    cap = cv2.VideoCapture(vid)

    if not cap.isOpened():
        print(f"{Cols.CRED}Error: Could not open video.{Cols.CEND}")
        exit()
    else:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"{Cols.CGREEN}Image size: {frame_width}:{frame_height} | Frames: {length} {Cols.CEND}")

        frame_start = Constants.FRAME_START 
        frame_stop = Constants.FRAME_STOP if (Constants.FRAME_STOP and (Constants.FRAME_STOP <= length)) else length

        # Process the video frame by frame
        for i in range(frame_start, frame_stop):
            aoi_name = ''
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect AprilTags in the grayscale image
            tags = detector.detect(gray)

            try:
                gaze_point = tuple(gaze.compr_data[i, [3, 4]])
                point = Gaze.transform_point_to_opencv(gaze_point, frame_width, frame_height)
                left, right = 0, 0
                # Select AprilTags
                tags_selected = {tag.tag_id: tag for tag in tags if tag.tag_id in tag_ids}
                #print("Tags IDs:", tags_selected.keys() )
                left = left_aoi.update(tags_selected, point)
                right = right_aoi.update(tags_selected, point)
                if left:
                    aoi_name = 'left'
                elif right:
                    aoi_name = 'right'
                else:
                    aoi_name = 'white-space'
                    white_space_aoi.gaze_sum += 1
                #white_space_aoi.gaze_sum = white_space_aoi.gaze_sum + 1 if not(left or right) else white_space_aoi.gaze_sum

                #TODO: Update the gaze data table in gaze object method
                to_update = {'aoi-left': str(left_aoi.vertices), 
                             'aoi-right': str(right_aoi.vertices), 
                             'aoi-gaze': aoi_name}
                # print(i, to_update)
                gaze.data_table.loc[gaze.data_table['world_index']==i, list(to_update.keys())] = to_update.values()
                

            except Exception as e:
                logging.error(f"Error processing frame {i}: {e}")
                # print(f"Error processing frame {i}: {e}")
                # exit()


    cap.release()
    cv2.destroyAllWindows()
    print(f"Left AOI: {left_aoi.gaze_sum} | Right AOI: {right_aoi.gaze_sum} | White space: {white_space_aoi.gaze_sum}")
    gaze.data_table.to_csv(f"{Constants.PATH_GAZE}{Constants.TABLE_GAZE}_{datetime_str}.csv", index=False)


def main(argv):
    if argv:
        
        # manage settings read form a yaml file
        if argv[0] == '--set':
            file_settings = argv[1]
            Settings.load_from_yaml(file_settings)
            settings = Settings()
            settings.update_dirs()
            
            print(settings.__dict__)
            run(vid=settings.VID, detector=detector, tag_ids=settings.TAGS['IDS'])
    else:
        print(Cols.CRED + "Use --set *.yml to load settings" + Cols.CEND, "\nThe end", sep='')
        sys.exit()


if __name__ == "__main__":
    main(sys.argv[1:])
