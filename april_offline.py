# ======================================================================= #
# analyze gaze data from Pupil Core eye tracker using apriltag detection  #
#                                                                         #
# use: python april_offline.py --set settings-6.yml --run [0, 1, 2, 3]    #
#      0: run analysis; 1: run aoi view; 2: run tags view; 3: run test    #
# ========================================================================#

import yaml
import sys
import os
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

# class Constants:
#     # april tags
#     TAGS = {
#             'FAMILY': "tag36h11",
#             'IDS': (20, 17, 22, 19, 23, 18),
#             'CORNERS': {20: 0, 17: 3, 22: 0.5, 19: 1.5, 23: 1, 18: 0},
#             }
#     Y_CORRECT = 0.25

#     COORD = {
#             0.5: (0, 1),
#             1.5: (0, 3)
#             }

#     # paths to video files and gaze data files
#     PATH_VID = "video/"
#     FILE = "world.mp4"
#     VID = PATH_VID + FILE # 0 for webcam
#     PATH_GAZE = "gaze/"
#     FILE_GAZE = "gaze_positions.csv"
#     GAZE = PATH_GAZE + FILE_GAZE
#     TABLE_GAZE = "gaze_table"
#     FRAME_START = 0
#     FRAME_STOP = None


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
    ALPHA: float = field(default=0.5)

    AOIS: List[Dict] = field(default_factory=list)

    # paths to video files and gaze data files
    PATH_VID: Any = field(default='')
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
        '''
        Update video and gaze file paths
        '''
        if self.PATH_VID == 0:
            self.VID = 0
        else:   
            if self.FILE:
                self.VID = self.PATH_VID + self.FILE
            else:
                print(f"{Cols.CRED}No video file path: *{self.FILE}* {Cols.CEND}")
                sys.exit()
        
        if self.FILE_GAZE:
            self.GAZE = self.PATH_GAZE + self.FILE_GAZE
        else:
            print(f"{Cols.CRED}No gaze file path{Cols.CEND}")
            sys.exit()
        


now = datetime.now()
datetime_str = now.strftime("%Y%m%d_%H%M%S")

# Configure logging
logging.basicConfig(filename='log/april_error.log',
                    level=logging.ERROR,
                    format='%(asctime)s %(levelname)s %(message)s')


@dataclass
class Aoi:
    '''
    Area of interest class
    '''
    settings: AppSettings
    name: str
    tag_upleft: Optional[Dict[int, Tuple[float, float]]|None] = None
    tag_upright: Optional[Dict[int, Tuple[float, float]]|None] = None
    tag_botleft: Optional[Dict[int, Tuple[float, float]]|None] = None
    tag_botright: Optional[Dict[int, Tuple[float, float]]|None] = None
    horiz: Optional[Tuple[float, float]|None] = None
    vert: Optional[Tuple[float, float]|None] = None

    tag_ids: Optional[List[int]|None] = None
    #pts: Optional[List[List[float]]]
    vertices = Optional[List[List[float|int]]|None]
    gaze_sum: int = 0


    def update(self, tags_selected: Dict, point: Tuple[int, int], draw: bool=False, frame: Optional[np.ndarray|None]=None) -> None:
        '''
        Update AOI with new tag detections
        '''
        if (len( tags_to_aoi := {tag.tag_id: tag for _, tag in tags_selected.items() if tag.tag_id in self.tag_ids} ) in (3, 4)):
            #print(f"AOI: {self.name}; tags: {tags_to_aoi.keys()}")
            pts = self._establish_pts(tags=tags_to_aoi, ids=self.tag_ids)
            if [None, None] in pts:
                pts.append(self._find_fourth_vertex(pts))
                pts.remove([None, None])
        
            self.vertices = self._sort_vertices(pts)

            if draw and frame is not None:
                self._draw_aoi(frame=frame)
            
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
    
    # TODO: zmienic uwzględniając nowe ustawienia
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
                    corner = self.settings.TAGS['CORNERS'][tag.tag_id]
                    if corner in (0, 1, 2, 3):
                        pt = list(tag.corners[self.settings.TAGS['CORNERS'][tag.tag_id]].astype(int))              
                        pt[1] = pt[1] + int(size * self.settings.Y_CORRECT) if tag.tag_id in (20, 22, 23) else pt[1] - int(size * self.settings.Y_CORRECT)
                    else:
                        pt = list((tag.center[0].astype(int), tag.corners[self.settings.COORD[corner][0]][1].astype(int)))
                        pt[1] = pt[1] + int(size * self.settings.Y_CORRECT) if tag.tag_id in (20, 22, 23) else pt[1] - int(size * self.settings.Y_CORRECT)

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
    
    def _draw_aoi(self, frame):
        '''
        Draws the quadrilateral (aoi)
        parameters:
        frame: image frame
        tags: dictionary of apriltag ids and their corresponding apriltag objects
        '''
        
        pts = self.vertices

        # for i, p in enumerate(pts, start=1):
        #     cv2.putText(frame, str(i), (p[0] + 10, p[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #     #cv2.circle(frame, (p[0], p[1]), 5, (0, 0, 255), -1)

        try:
            pts_poly = np.array(pts, np.int32).reshape((-1, 1, 2))
            isClosed = True
            color = (255, 50, 50)
            thickness = 2
            cv2.polylines(frame, [pts_poly], isClosed, color, thickness)
            return pts #transform_vertices_to_opencv(pts, frame_width, frame_height)
        except Exception as e:
            logging.error(f"Error drawing aoi: {e}")
            # print(f"Error drawing aoi: {e}")
            # exit()
            return None



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
            self.data = np.genfromtxt(self.path, delimiter=",", dtype=float, skip_header=1, skip_footer=1)
            index_column = 1
            # Get unique indices and their first occurrence positions
            _, unique_pos = np.unique(self.data[:, index_column], return_index=True)
            # Select the rows corresponding to the first occurrence of each unique index
            self.compr_data = self.data[unique_pos]
            # Pandas table
            self.data_table = pd.read_csv(self.path, sep=",")
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
    
    # TODO: Update the table with the aoi data
    def update_table(self, frame: int, to_update: Dict) -> None:
        self.gaze.data_table.loc[self.gaze.data_table['world_index']==frame, list(to_update.keys())] = to_update.values()


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

class Runner:
    '''
    Class to run visualizations and analysis 
    '''

    def __init__(self, settings, detector) -> None:
        self.settings = settings
        self.detector = detector

        self.vid = settings.VID 
        self.tag_ids = settings.TAGS['IDS']
        self.aois = []#settings.AOIS
        self.gaze = Gaze(path=settings.GAZE)
        # self.left_aoi = Aoi(name="left", settings=settings, tag_ids=self.tag_ids[:4])
        # self.right_aoi = Aoi(name="right", settings=settings, tag_ids=self.tag_ids[2:6])
        for aoi in settings.AOIS:
            aoi_tag_ids = [val['id'] for key, val in aoi.items() if key in ('tag_upleft', 'tag_upright', 'tag_botleft', 'tag_botright')]
            self.aois.append(Aoi(name=aoi['name'], 
                                 tag_upleft=aoi['tag_upleft'],
                                 tag_upright=aoi['tag_upright'],
                                 tag_botleft=aoi['tag_botleft'],
                                 tag_botright=aoi['tag_botright'],
                                 horiz=aoi['horiz'],
                                 vert=aoi['vert'],
                                 settings=settings, 
                                 tag_ids=aoi_tag_ids))
            
        self.white_space_aoi = Aoi(name="white-space", settings=settings)
        

    def run_analysis(self) -> None:
        '''
        Main function to run apriltag detection
        parameters:
        vid: video file path; if 0, webcam will be used
        '''
        print("I am running the analysis")
        #exit()
        cap = cv2.VideoCapture(self.vid)

        if not cap.isOpened():
            print(f"{Cols.CRED}Error: Could not open video.{Cols.CEND}")
            exit()
        else:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"{Cols.CGREEN}Image size: {frame_width}:{frame_height} | Frames: {length} {Cols.CEND}")

            frame_start = self.settings.FRAME_START 
            frame_stop = self.settings.FRAME_STOP if (self.settings.FRAME_STOP and (self.settings.FRAME_STOP <= length)) else length

            # Process the video frame by frame
            for i in range(frame_start, frame_stop):
                aoi_name = ''
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect AprilTags in the grayscale image
                tags = self.detector.detect(gray)

                try:
                    gaze_point = tuple(self.gaze.compr_data[i, [3, 4]])
                    point = Gaze.transform_point_to_opencv(gaze_point, frame_width, frame_height)
                    left, right = 0, 0
                    # Select AprilTags
                    tags_selected = {tag.tag_id: tag for tag in tags if tag.tag_id in self.tag_ids}
                    #print("Tags IDs:", tags_selected.keys() )
                    left = self.left_aoi.update(tags_selected, point)
                    right = self.right_aoi.update(tags_selected, point)
                    if left:
                        aoi_name = 'left'
                    elif right:
                        aoi_name = 'right'
                    else:
                        aoi_name = 'white-space'
                        self.white_space_aoi.gaze_sum += 1
                    #white_space_aoi.gaze_sum = white_space_aoi.gaze_sum + 1 if not(left or right) else white_space_aoi.gaze_sum

                    
                    to_update = {'aoi-left': str(self.left_aoi.vertices), 
                                'aoi-right': str(self.right_aoi.vertices), 
                                'aoi-gaze': aoi_name}
                    # print(i, to_update)
                    self.gaze.update_table(i, to_update)
                    #self.gaze.data_table.loc[self.gaze.data_table['world_index']==i, list(to_update.keys())] = to_update.values()
                    

                except Exception as e:
                    logging.error(f"Error processing frame {i}: {e}")
                    # print(f"Error processing frame {i}: {e}")
                    # exit()


        cap.release()
        cv2.destroyAllWindows()
        print(f"Left AOI: {self.left_aoi.gaze_sum} | Right AOI: {self.right_aoi.gaze_sum} | White space: {self.white_space_aoi.gaze_sum}")
        self.gaze.data_table.to_csv(f"{self.settings.PATH_GAZE}{self.settings.TABLE_GAZE}_{datetime_str}.csv", index=False)

    
    
    def run_view_aoi(self) -> None:
        '''
        View the detected AOIs in the video
        '''
        print("I am running AOI detection")

        num_gaze_left = 0
        num_gaze_right = 0
        
        # if self.vid:

        cap = cv2.VideoCapture(self.vid)

        if not cap.isOpened():
            print(f"{Cols.CRED}Error: Could not open video.{Cols.CEND}")
            exit()
        else:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.vid else 0
            print(f"{Cols.CGREEN}Image size: {frame_width}:{frame_height} | Frames: {length} | START STREAMING - Press Q to exit{Cols.CEND}")

        num_frame = 0

        while cap.isOpened():
            # if self.vid:
            

            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to grayscale as the detector requires a single channel image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect AprilTags in the grayscale image
            tags = self.detector.detect(gray)

            # Draw bounding boxes around detected AprilTags
            tags_selected = {tag.tag_id: tag for tag in tags if tag.tag_id in self.tag_ids} if self.vid else {tag.tag_id: tag for tag in tags}

            # if len(tags_0_3) == 4:
            #     pprint(tags_0_3.keys())
            #     break

        
            for _, tag in tags_selected.items():
                if self.vid:
                    corner = self.settings.TAGS['CORNERS'][tag.tag_id]
                    if corner in (0, 1, 2, 3):
                        pt = tuple(tag.corners[self.settings.TAGS['CORNERS'][tag.tag_id]].astype(int))
                    else:
                        pt = (tag.center[0].astype(int), tag.corners[self.settings.COORD[corner][0]][1].astype(int))
                    
                    cv2.circle(frame, pt, 5, (255, 255, 0), -1)

                    for idx in range(len(tag.corners)):
                        
                        pt1 = tuple(tag.corners[idx].astype(int))
                        pt2 = tuple(tag.corners[(idx + 1) % len(tag.corners)].astype(int))
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

                # Optionally, draw the tag ID
                tag_id = str(tag.tag_id)
                cv2.putText(frame, tag_id, (tag.center[0].astype(int), tag.center[1].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
            # *****
            try:
                gaze_point = tuple(self.gaze.compr_data[num_frame, [3, 4]])
                point = Gaze.transform_point_to_opencv(gaze_point, frame_width, frame_height)
                left, right = 0, 0
                # Select AprilTags
                tags_selected = {tag.tag_id: tag for tag in tags if tag.tag_id in self.tag_ids}
                #print("Tags IDs:", tags_selected.keys() )
                left = self.left_aoi.update(tags_selected, point, draw=True, frame=frame)
                right = self.right_aoi.update(tags_selected, point, draw=True, frame=frame)
                if left:
                    aoi_name = 'left'
                elif right:
                    aoi_name = 'right'
                else:
                    aoi_name = 'white-space'
                    self.white_space_aoi.gaze_sum += 1
                
                

            except Exception as e:
                # logging.error(f"Error processing frame {num_frame}: {e}")
                # print(f"Error processing frame {frame}: {e}")
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(f"Error processing frame {frame}: {e}, {exc_type}, {fname}, {exc_tb.tb_lineno}")
                exit()
            

            # *****
            if self.vid:
                overlay = frame.copy()
                cv2.circle(overlay, point, 15, (0, 0, 255), -1)

            blended_image = cv2.addWeighted(overlay, self.settings.ALPHA, frame, 1-self.settings.ALPHA, 0) if self.vid else frame
            # Display the frame with detected tags
            cv2.imshow('Pupil Labs AprilTag Detection', blended_image)


            # Press 'q' to exit the video display loop
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            
            num_frame += 1
            # if num_frame >= 10:
            #     break
        # Release resources
        print(f"Frames read: {num_frame}")
        print(f"{Cols.CGREEN}Left: {num_gaze_left} | Right: {num_gaze_right}{Cols.CEND}")
        cap.release()
        cv2.destroyAllWindows()




    def run_view_tags(self) -> None:
        '''
        View the detected tags in the video
        '''
        print("I am running tags detection")

        cap = cv2.VideoCapture(self.vid)

        if not cap.isOpened():
            print(f"{Cols.CRED}Error: Could not open video.{Cols.CEND}")
            exit()
        else:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"{Cols.CGREEN}Image size: {frame_width}:{frame_height} | Frames: {length} {Cols.CEND}")

            num_frame = 0

        while cap.isOpened():
            
            ret, frame = cap.read()
            if not ret:
                break

            
            # Convert the frame to grayscale as the detector requires a single channel image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect AprilTags in the grayscale image
            tags = self.detector.detect(gray)

            # Draw bounding boxes around detected AprilTags
            tags_selected = {tag.tag_id: tag for tag in tags}

            
            for _, tag in tags_selected.items():
                for idx in range(len(tag.corners)):
                    
                    pt1 = tuple(tag.corners[idx].astype(int))
                    pt2 = tuple(tag.corners[(idx + 1) % len(tag.corners)].astype(int))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                # Optionally, draw the tag ID
                tag_id = str(tag.tag_id)
                cv2.putText(frame, tag_id, (tag.center[0].astype(int), tag.center[1].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow('Pupil Labs AprilTag Detection', frame)


            # Press 'q' to exit the video display loop
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            
            num_frame += 1
            # if num_frame >= 1200:
            #     break
        # Release resources
        print(f"Frames read: {num_frame}")
        cap.release()
        cv2.destroyAllWindows()
            

    def run_test(self) -> None:
        '''
        Test AOIS
        '''
        print("I am running AOIS test")
        
        for aoi in self.aois:
            pprint(aoi.__dict__, sort_dicts=False)

        sys.exit()

        cap = cv2.VideoCapture(self.vid)

        if not cap.isOpened():
            print(f"{Cols.CRED}Error: Could not open video.{Cols.CEND}")
            exit()
        else:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.vid else 0
            print(f"{Cols.CGREEN}Image size: {frame_width}:{frame_height} | Frames: {length} | START STREAMING - Press Q to exit{Cols.CEND}")

        num_frame = 0

        while cap.isOpened():
            # if self.vid:
            

            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to grayscale as the detector requires a single channel image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect AprilTags in the grayscale image
            tags = self.detector.detect(gray)

            # Draw bounding boxes around detected AprilTags
            # ********* TODO: dokonczyc wyswietlanie AOI na podstawie nowych ustawien
            #tags_selected = {tag.tag_id: tag for tag in tags if tag.tag_id in self.tag_ids} if self.vid else {tag.tag_id: tag for tag in tags}
            


            try:
                gaze_point = tuple(self.gaze.compr_data[num_frame, [3, 4]])
                point = Gaze.transform_point_to_opencv(gaze_point, frame_width, frame_height)
                #left, right = 0, 0
                # Select AprilTags
                tags_selected = {tag.tag_id: tag for tag in tags if tag.tag_id in self.tag_ids}
                
                #print("Tags IDs:", tags_selected.keys() )

                for aoi in self.aois:
                    aoi.update(tags_selected, point, draw=True, frame=frame)

                # left = self.left_aoi.update(tags_selected, point, draw=True, frame=frame)
                # right = self.right_aoi.update(tags_selected, point, draw=True, frame=frame)
                # if left:
                #     aoi_name = 'left'
                # elif right:
                #     aoi_name = 'right'
                # else:
                #     aoi_name = 'white-space'
                #     self.white_space_aoi.gaze_sum += 1
                
                

            except Exception as e:
                # logging.error(f"Error processing frame {num_frame}: {e}")
                # print(f"Error processing frame {frame}: {e}")
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(f"Error processing frame {frame}: {e}, {exc_type}, {fname}, {exc_tb.tb_lineno}")
                exit()
            

            # *****
            if self.vid:
                overlay = frame.copy()
                cv2.circle(overlay, point, 15, (0, 0, 255), -1)

            blended_image = cv2.addWeighted(overlay, self.settings.ALPHA, frame, 1-self.settings.ALPHA, 0) if self.vid else frame
            # Display the frame with detected tags
            cv2.imshow('Pupil Labs AOI Test', blended_image)

            #cv2.imshow('Pupil Labs AOI Test', frame)

            # Press 'q' to exit the video display loop
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            
            num_frame += 1
            # if num_frame >= 10:
            #     break
        # Release resources
        print(f"Frames read: {num_frame}")
        cap.release()
        cv2.destroyAllWindows()



def main(argv):
    if argv:
        print(argv)
        
        # manage settings read form a yaml file
        if argv[0] == '--set':
            file_settings = argv[1]
            Settings.load_from_yaml(file_settings)
            settings = Settings()
            settings.update_dirs()
            # configure april tag detector
            detector = Detector(
                families=settings.TAGS['FAMILY'],
                nthreads=1,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
                debug=0
            )
            runner = Runner(settings, detector=detector)
            #pprint(settings.__dict__)
            # pprint(runner.aois)
            # sys.exit()
            if argv[2] == '--run':
                if argv[3] == '0':
                    runner.run_analysis()
                elif argv[3] == '1':
                    runner.run_view_aoi()
                elif argv[3] == '2':
                    runner.run_view_tags()
                elif argv[3] == '3':
                    runner.run_test()
                else:
                    print(f"{Cols.CRED}Error: Invalid argument {argv[3]}{Cols.CEND}")
                    sys.exit()
            else:
                runner.run_analysis()
    else:
        print(Cols.CRED + "Use --set *.yml to load settings" + Cols.CEND, "\nThe end", sep='')
        sys.exit()


if __name__ == "__main__":
    main(sys.argv[1:])
