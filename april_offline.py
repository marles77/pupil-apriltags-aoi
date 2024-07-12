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
import progressbar


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


@dataclass
class AppSettings:
    """
    The AppSettings class.
    """

    TAGS: defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))
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
    color_bgr: Optional[Tuple[int, int, int]|None] = None

    tag_ids: Optional[List[int]|None] = None
    vertices = Optional[List[List[float|int]]|None]
    gaze_sum: int = 0


    def update(self, tags_selected: Dict, point: Tuple[int, int], draw: bool=False, frame: Optional[np.ndarray|None]=None) -> None:
        '''
        Update AOI with new tag detections
        '''

        if (len( tags_to_aoi := {tag.tag_id: tag for _, tag in tags_selected.items() if tag.tag_id in self.tag_ids} ) in (3, 4)):
            #print(f"AOI: {self.name}; tags: {tags_to_aoi.keys()}")
            pts = self._establish_pts(tags=tags_to_aoi, ids=self.tag_ids)
            
            pts_sorted = self._sort_vertices(pts)
            self.vertices = self._establish_vertices(pts_sorted)

            if draw and frame is not None:
                self._draw_aoi(frame=frame)
            
            if Gaze.is_point_in_quadrilateral(point[0], point[1], self.vertices):
                self.gaze_sum += 1
                return 1
            else:
                return 0
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
        return vertices
    
    
    def _establish_pts(self, tags: Dict, ids: Tuple) -> List[List[float]]:
        '''
        Establishes the vertices of the quadrilateral (defined by the apriltags)
        parameters:
            tags: dictionary of apriltag ids and their corresponding apriltag objects
            ids: tuple of apriltag ids
        '''

        pts = []
        ids_sorted = [tags.get(id, None) for id in ids]
        aoi_tags = (self.tag_upleft, self.tag_upright, self.tag_botright, self.tag_botleft)

        try:   
            for i, current_tag in enumerate(ids_sorted, start=1):
                pt = [None, None]
                if current_tag is None:
                    pt = [None, None]
                    
                else:
                    try:
                        tag_width = np.abs(current_tag.corners[0][0] - current_tag.corners[2][0])
                        
                    except Exception as e:
                        logging.error(f"Error getting width: {e}, corner: {type(id.corners)}")
                    
                    try:
                        tag_height = abs(current_tag.corners[0][1] - current_tag.corners[2][1])
                        
                    except Exception as e:
                        logging.error(f"Error getting height: {e}")
                    
                    try:
                        center = [current_tag.center[0], current_tag.center[1]]
                        
                    except Exception as e:
                        logging.error(f"Error getting center: {e}")
                    
                    try:
                        pt = [[int(center[0] + (tag_width/2)*tag['xy'][0]), int(center[1] + (tag_height/2)*tag['xy'][1])] 
                              for tag in aoi_tags if current_tag.tag_id==tag['id']][0]
                        
                    except Exception as e:
                        logging.error(f"Error getting point: {e}")
                    

                pts.append(pt)
            
            if [None, None] in pts:
                pts.append(self._find_fourth_vertex(pts))
                pts.remove([None, None])
                  
        except Exception as e:
            logging.error(f"Error establishing pts: {e}")
            return pts
        return pts
    

    def _establish_vertices(self, pts: List[List[float]]) -> List[Tuple[float, float]]:
        '''
        Establishes the vertices of the quadrilateral (AOI)
        '''

        width_upper = abs(pts[0][0] - pts[1][0])
        width_lower = abs(pts[2][0] - pts[3][0])
        height_left = abs(pts[0][1] - pts[3][1])
        height_right = abs(pts[1][1] - pts[2][1])

        res_pts = [None, None, None, None]

        res_pts[0] = [int(pts[0][0] + width_upper * self.horiz[0]), int(pts[0][1] + height_left * self.vert[0])]
        res_pts[1] = [int(pts[1][0] - width_upper * (1-self.horiz[1])), int(pts[1][1] + height_left * (self.vert[0]))]
        res_pts[2] = [int(pts[2][0] - width_lower * (1-self.horiz[1])), int(pts[2][1] - height_right * (1-self.vert[1]))]
        res_pts[3] = [int(pts[3][0] + width_lower * self.horiz[0]), int(pts[3][1] - height_right * (1-self.vert[1]))]
        
        return res_pts
    

    def _transform_vertices_to_opencv(self, vertices: List[Tuple[float, float]], frame_width: int, frame_height: int) -> List[Tuple[int, int]]:
        """
        Transform the vertices of the quadrilateral to the OpenCV coordinate system.
        parameters:
            vertices: List of tuples [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] in the original coordinate system
            frame_width: Width of the OpenCV frame
            frame_height: Height of the OpenCV frame
        returns: 
            List of tuples in the OpenCV coordinate system
        """
        return [Gaze.transform_point_to_opencv(vertex, frame_width, frame_height) for vertex in vertices]
    

    def _draw_aoi(self, frame) -> None:
        '''
        Draws the quadrilateral (aoi)
        parameters:
            frame: image frame
        '''
        
        pts = self.vertices

        try:
            pts_poly = np.array(pts, np.int32).reshape((-1, 1, 2))
            isClosed = True
            color = tuple(self.color_bgr) #(255, 50, 50)
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
    Class to store and process gaze data
    '''

    path: str
    data: np.ndarray = None
    compr_data: np.ndarray = None # Compressed gaze data - to match world camera frames
    data_table: pd.DataFrame = None # Pandas table to store the gaze data and aoi data
    aois_names: List[str] = field(default_factory=list)
    
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
            # Prepare new columns for the aoi data
            for name in self.aois_names:
                self.data_table[f"{name}"] = [None] * self.data_table.shape[0]
            
            self.data_table['aoi-gaze'] = [''] * self.data_table.shape[0]
            
            print(f"{Cols.CYELLOW}Compressed Rows: {self.compr_data.shape[0]} | 1st: {self.compr_data[0, 1]} | Last: {self.compr_data[-1, 1]} | DataFrame: {self.data_table.shape}{Cols.CEND}")
            
        except Exception as e:
            logging.error(f"Error reading gaze data: {e}")
            print(f"Error reading gaze data: {e}")
            sys.exit()
        finally:
            print(f"{Cols.CGREEN}Gaze data loaded | Rows: {self.data.shape[0]}{Cols.CEND}")
            
    
    def update_table(self, frame: int, to_update: Dict) -> None:
        '''
        Update the table with the aoi data
        '''

        try:
            self.data_table.loc[self.data_table['world_index']==frame, list(to_update.keys())] = to_update.values()
        except Exception as e:
            logging.error(f"Error updating table: {e}")


    @staticmethod
    def transform_point_to_opencv(point: Tuple[float, float], frame_width: int, frame_height: int) -> Tuple[int, int]:
        """
        Transform a point from the original coordinate system to the OpenCV coordinate system.
        parameters: 
            point: Tuple (x, y) in the original coordinate system (0,0 to 1,1)
            frame_width: Width of the OpenCV frame
            frame_height: Height of the OpenCV frame
        returns: 
            Tuple (new_x, new_y) in the OpenCV coordinate system
        """

        point_x, point_y = point
        new_x = int(point_x * frame_width)
        new_y = int((1 - point_y) * frame_height)
        return new_x, new_y
    

    @staticmethod
    def is_point_in_quadrilateral(px: int, py: int, vertices: List[Tuple[int, int]]) -> bool:
        """
        Determine if the point (px, py) is inside the quadrilateral defined by vertices.
        parameters: 
            px: x coordinate of the point
            py: y coordinate of the point
            vertices: List of tuples [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        returns: 
            True if the point is inside the quadrilateral, False otherwise
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
        self.aois = []
        
        # Create the AOI objects
        for aoi in settings.AOIS:
            aoi_tag_ids = [val['id'] for key, val in aoi.items() if key in ('tag_upleft', 'tag_upright', 'tag_botleft', 'tag_botright')]
            self.aois.append(Aoi(name=aoi['name'], 
                                 tag_upleft=aoi['tag_upleft'],
                                 tag_upright=aoi['tag_upright'],
                                 tag_botleft=aoi['tag_botleft'],
                                 tag_botright=aoi['tag_botright'],
                                 horiz=aoi['horiz'],
                                 vert=aoi['vert'],
                                 color_bgr=aoi['color_bgr'],
                                 settings=settings, 
                                 tag_ids=aoi_tag_ids))
            
        self.white_space_aoi = Aoi(name="white-space", settings=settings)
        self.gaze = Gaze(path=settings.GAZE, aois_names=[aoi.name for aoi in self.aois])


    def run_analysis(self) -> None:
        '''
        Main function to run analysis of gaze data in reference to the AOIs
        '''

        print("I am running the analysis")

        if self.vid == 0:
            print(f"{Cols.CRED}Error: Webcam not supported in the analysis mode.{Cols.CEND}")
            sys.exit()

        cap = cv2.VideoCapture(self.vid)

        if not cap.isOpened():
            print(f"{Cols.CRED}Error: Could not open video.{Cols.CEND}")
            exit()
        else:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            start_frame = self.settings.FRAME_START 
            stop_frame = self.settings.FRAME_STOP if (self.settings.FRAME_STOP and (self.settings.FRAME_STOP <= length)) else length

            total_frames = stop_frame - start_frame
            print(f"{Cols.CGREEN}Image size: {frame_width}:{frame_height} | Frames: {length} | Frames to analyze: {total_frames}{Cols.CEND}")

            bar = progressbar.ProgressBar(max_value=total_frames)

            # Process the video frame by frame
            for i in range(start_frame, stop_frame):
                bar.update(i-start_frame)
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
                    
                    # Select AprilTags
                    tags_selected = {tag.tag_id: tag for tag in tags if tag.tag_id in self.tag_ids}

                    for aoi in self.aois:
                        res = aoi.update(tags_selected, point)
                        aoi_name = aoi_name + aoi.name + ' ' if res else aoi_name

                    aoi_name = aoi_name if aoi_name else 'white-space'

                    to_update = {aoi.name: str(aoi.vertices) for aoi in self.aois}
                    to_update['aoi-gaze'] = aoi_name
                    
                    self.gaze.update_table(i, to_update)  

                except Exception as e:
                    logging.error(f"Error processing frame {i}: {e}")
                    
        cap.release()
        cv2.destroyAllWindows()
        
        # Save the data to table
        self.gaze.data_table.to_csv(f"{self.settings.PATH_GAZE}{self.settings.TABLE_GAZE}_{datetime_str}.csv", index=False)

    
    def run_view_aoi(self) -> None:
        '''
        View the detected AOIs in the video along with the gaze data
        '''

        print("I am running AOI detection")

        cap = cv2.VideoCapture(self.vid)

        if not cap.isOpened():
            print(f"{Cols.CRED}Error: Could not open video.{Cols.CEND}")
            sys.exit()
        else:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"{Cols.CGREEN}Image size: {frame_width}:{frame_height} | Frames: {length} | START STREAMING - Press Q to exit{Cols.CEND}")
            if length <= self.settings.FRAME_START:
                print(f"{Cols.CRED}Error: Video is too short. Check your settings file (FRAME_START is set at {self.settings.FRAME_START}).{Cols.CEND}")
                sys.exit()

        start_frame = self.settings.FRAME_START if self.vid != 0 else 0 # start from 0 if webcam is used
        stop_frame = self.settings.FRAME_STOP if self.settings.FRAME_STOP != None else length
        num_frame = start_frame

        total_frames = stop_frame - start_frame
        bar = progressbar.ProgressBar(max_value=total_frames)
        
        # set the video position to the start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        i = 0

        while cap.isOpened():
            bar.update(i)
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to grayscale as the detector requires a single channel image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect AprilTags in the grayscale image
            tags = self.detector.detect(gray)

            try:
                gaze_point = tuple(self.gaze.compr_data[num_frame, [3, 4]])
                point = Gaze.transform_point_to_opencv(gaze_point, frame_width, frame_height)
                #left, right = 0, 0

                for aoi in self.aois:
                    # Select AprilTags
                    try:
                        tags_selected = {tag.tag_id: tag for tag in tags if tag.tag_id in aoi.tag_ids}
                        aoi.update(tags_selected, point, draw=True, frame=frame)
                    except:
                        print("Błąd przy update")
                        exit()
                
            except Exception as e:
                logging.error(f"Error processing frame {num_frame}: {e}")
                # print(f"Error processing frame {frame}: {e}")
                # exc_type, exc_obj, exc_tb = sys.exc_info()
                # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                # print(f"Error processing frame {frame}: {e}, {exc_type}, {fname}, {exc_tb.tb_lineno}")
                # sys.exit()
            
            if self.vid:
                overlay = frame.copy()
                cv2.circle(overlay, point, 15, (0, 0, 255), -1)

            blended_image = cv2.addWeighted(overlay, self.settings.ALPHA, frame, 1-self.settings.ALPHA, 0) if self.vid else frame
            # Display frame number
            cv2.rectangle(blended_image, (10, 5), (80, 28), (0, 0, 0), -1)
            cv2.putText(blended_image, str(num_frame), (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            # Display the frame
            cv2.imshow('Pupil Labs AprilTag Detection', blended_image)


            # Press 'q' to exit the video display loop
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            
            num_frame += 1
            i += 1
            if num_frame >= stop_frame:
                break
        
        # Release resources
        print(f"\nFrames read: {num_frame}")
        print("AOI gaze distribution:")

        for aoi in self.aois:
            print(f"AOI {aoi.name}: {aoi.gaze_sum}")
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
            print(f"{Cols.CGREEN}Image size: {frame_width}:{frame_height} | Frames: {length} | START STREAMING - Press Q to exit{Cols.CEND}")
            if length <= self.settings.FRAME_START:
                print(f"{Cols.CRED}Error: Video is too short. Check your settings file (FRAME_START is set at {self.settings.FRAME_START}).{Cols.CEND}")
                exit()

        start_frame = self.settings.FRAME_START if self.vid != 0 else 0 # start from 0 if webcam is used
        stop_frame = self.settings.FRAME_STOP if self.settings.FRAME_STOP != None else length
        num_frame = start_frame
        
        # set the video position to the start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

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

            # loop over the detected tags and put labels
            for _, tag in tags_selected.items():
                if num_frame == 0:
                    print(f"Tag identified: {tag.tag_id}")
                
                for idx in range(len(tag.corners)):
                    
                    pt1 = tuple(tag.corners[idx].astype(int))
                    pt2 = tuple(tag.corners[(idx + 1) % len(tag.corners)].astype(int))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                # Optionally, draw the tag ID
                tag_id = str(tag.tag_id)
                cv2.rectangle(frame, (tag.center[0].astype(int) - 15, tag.center[1].astype(int) - 15), 
                              (tag.center[0].astype(int) + 15, tag.center[1].astype(int) + 15), (50, 50, 50), -1)
                cv2.putText(frame, tag_id, (tag.center[0].astype(int)-10, tag.center[1].astype(int)+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
            # Display frame number
            cv2.rectangle(frame, (10, 5), (80, 28), (0, 0, 0), -1)
            cv2.putText(frame, str(num_frame), (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.imshow('Pupil Labs AprilTag Detection', frame)

            # Press 'q' to exit the video display loop
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            
            num_frame += 1
            if num_frame >= stop_frame:
                break
        
        # Release resources
        print(f"Frames read: {num_frame - start_frame}")
        cap.release()
        cv2.destroyAllWindows()
            

    def run_test(self) -> None:
        '''
        Test AOIS
        '''

        print("I am running AOIS test")
        
        for aoi in self.aois:
            pprint(aoi.__dict__, sort_dicts=False)

        cap = cv2.VideoCapture(self.vid)

        if not cap.isOpened():
            print(f"{Cols.CRED}Error: Could not open video.{Cols.CEND}")
            sys.exit()
        else:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"{Cols.CGREEN}Image size: {frame_width}:{frame_height} | Frames: {length} | START STREAMING - Press Q to exit{Cols.CEND}")
            if length <= self.settings.FRAME_START:
                print(f"{Cols.CRED}Error: Video is too short. Check your settings file (FRAME_START is set at {self.settings.FRAME_START}).{Cols.CEND}")
                sys.exit()

        start_frame = self.settings.FRAME_START if self.vid != 0 else 0 # start from 0 if webcam is used
        stop_frame = self.settings.FRAME_STOP if self.settings.FRAME_STOP != None else length
        num_frame = start_frame
        
        # set the video position to the start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to grayscale as the detector requires a single channel image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect AprilTags in the grayscale image
            tags = self.detector.detect(gray)

            try:
                gaze_point = tuple(self.gaze.compr_data[num_frame, [3, 4]])
                point = Gaze.transform_point_to_opencv(gaze_point, frame_width, frame_height)

                for aoi in self.aois:
                    # Select AprilTags
                    tags_selected = {tag.tag_id: tag for tag in tags if tag.tag_id in aoi.tag_ids}
                    aoi.update(tags_selected, point, draw=True, frame=frame)
                    
                    
            except Exception as e:
                logging.error(f"Error processing frame {num_frame}: {e}")
                # print(f"Error processing frame {frame}: {e}")
                # exc_type, exc_obj, exc_tb = sys.exc_info()
                # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                # print(f"Error processing frame {frame}: {e}, {exc_type}, {fname}, {exc_tb.tb_lineno}")
                # sys.exit()
            
            if self.vid:
                overlay = frame.copy()
                cv2.circle(overlay, point, 15, (0, 0, 255), -1)

            blended_image = cv2.addWeighted(overlay, self.settings.ALPHA, frame, 1-self.settings.ALPHA, 0) if self.vid else frame
            # Display frame number
            cv2.rectangle(blended_image, (10, 5), (80, 28), (0, 0, 0), -1)
            cv2.putText(blended_image, str(num_frame), (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            # Display the frame
            cv2.imshow('Pupil Labs AOI Test', blended_image)

            # Press 'q' to exit the video display loop
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            
            num_frame += 1
            if num_frame >= stop_frame:
                break
        
        # Release resources
        print(f"Frames read: {num_frame}")
        cap.release()
        cv2.destroyAllWindows()


def main(argv) -> None:
    '''
    Main function
    '''

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
