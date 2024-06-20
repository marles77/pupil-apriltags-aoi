# detect april tags

from pupil_apriltags import Detector
import cv2
import numpy as np
from pprint import pprint
import logging

CRED = '\033[91m'
CGREEN  = '\33[32m'
CYELLOW = '\33[33m'
CEND = '\033[0m'

TAGS = {
        'IDS': (20, 17, 22, 19, 23, 18),
        'CORNERS': {20: 0, 17: 3, 22: 0.5, 19: 1.5, 23: 1, 18: 0},
        }

COORD = {
         0.5: (0, 1),
         1.5: (0, 3)
         }

Y_CORRECT = 0.25

PATH_VID = "video/"
FILE = "world.mp4"
VID = 0 #PATH_VID + FILE # 0 for webcam
PATH_GAZE = "gaze/"
FILE_GAZE = "gaze_positions.csv"
GAZE = PATH_GAZE + FILE_GAZE

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)] # B, G, R, A
ALPHA = 0.5

AOI = False

# Configure logging
logging.basicConfig(filename='log/april_error.log',
                    level=logging.ERROR,
                    format='%(asctime)s %(levelname)s %(message)s')

detector = Detector(
   families="tag36h11",
   nthreads=1,
   quad_decimate=1.0,
   quad_sigma=0.0,
   refine_edges=1,
   decode_sharpening=0.25,
   debug=0
)


def find_fourth_vertex(pts):
    '''
    Finds the fourth vertex of the quadrilateral (aoi)
    parameters:
        pts: list of points of the quadrilateral
    returns:
        pt: [list] fourth vertex of the quadrilateral
    '''
    pt =[None, None]
    try:
        # tags_sorted = [tags.get(id, None) for id in TAGS['IDS']] 
        # pts = [ [tag.center[0].astype(int), tag.center[1].astype(int)] if tag != None else [None, None] for tag in tags_sorted]
        #tag.center[0].astype(int), tag.center[1].astype(int)] for id in IDS] 
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
        #pts = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    except Exception as e:
        logging.error(f"Error finding fourth vertex: {e}")
        # print(f"Error finding fourth vertex: {e}")
        # exit()
        
    return pt


def sort_vertices(vertices):
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


def establish_pts(tags, ids):
    '''
    Establishes the vertices of the quadrilateral (aoi)
    parameters:
    tags: dictionary of apriltag ids and their corresponding apriltag objects
    '''
    
    pts = []
    #tags_sorted = [tags.get(id, None) for id in TAGS['IDS']]
    tags_sorted = [tags.get(id, None) for id in ids]

    try:
        # for _, tag in tags.items():
        #     corner = TAGS['CORNERS'][tag.tag_id]
        #     if corner in (0, 1, 2, 3):
        #         pt = tuple(tag.corners[TAGS['CORNERS'][tag.tag_id]].astype(int))
        #     else:
        #         pt = (tag.center[0].astype(int), tag.corners[COORD[corner][0]][1].astype(int))
        #     pts.append(list(pt))
        for tag in tags_sorted:
            if tag == None:
                pt = (None, None)
            else:
                size = abs(tag.corners[0][1] - tag.corners[2][1])
                corner = TAGS['CORNERS'][tag.tag_id]
                if corner in (0, 1, 2, 3):
                    pt = list(tag.corners[TAGS['CORNERS'][tag.tag_id]].astype(int))              
                    pt[1] = pt[1] + int(size * Y_CORRECT) if tag.tag_id in (20, 22, 23) else pt[1] - int(size * Y_CORRECT)
                else:
                    pt = list((tag.center[0].astype(int), tag.corners[COORD[corner][0]][1].astype(int)))
                    pt[1] = pt[1] + int(size * Y_CORRECT) if tag.tag_id in (20, 22, 23) else pt[1] - int(size * Y_CORRECT)

            pts.append(pt)
    except Exception as e:
        logging.error(f"Error establishing pts: {e}")
        return pts
        # print(f"Error establishing pts: {e}")
        # exit()
    return pts

def transform_vertices_to_opencv(vertices, frame_width, frame_height):
    """
    Transform the vertices of the quadrilateral to the OpenCV coordinate system.

    :param vertices: List of tuples [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] in the original coordinate system
    :param frame_width: Width of the OpenCV frame
    :param frame_height: Height of the OpenCV frame
    :return: List of tuples in the OpenCV coordinate system
    """
    return [transform_point_to_opencv(vertex, frame_width, frame_height) for vertex in vertices]


def draw_aoi(frame, frame_width, frame_height, tags, ids):
    '''
    Draws the quadrilateral (aoi)
    parameters:
    frame: image frame
    tags: dictionary of apriltag ids and their corresponding apriltag objects
    '''
    
    pts = establish_pts(tags, ids)
    if [None, None] in pts:
        pts.append(find_fourth_vertex(pts))
        pts.remove([None, None])
    
    pts = sort_vertices(pts)

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
    


def transform_point_to_opencv(point, frame_width, frame_height):
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


def is_point_in_quadrilateral(px, py, vertices, num_frame):
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

def main(vid=VID, detector=detector, tag_ids=TAGS['IDS'], aoi=False):
    '''
    Main function to run apriltag detection
    parameters:
    vid: video file path; if 0, webcam will be used
    '''
    num_gaze_left = 0
    num_gaze_right = 0
    if aoi:
        try:
            gaze_data = np.genfromtxt(GAZE, delimiter=",", dtype=float, skip_header=1, skip_footer=1)
        except Exception as e:
            logging.error(f"Error reading gaze data: {e}")
            # print(f"Error reading gaze data: {e}")
            # exit()
        finally:
            print(f"{CGREEN}Gaze data loaded | Rows: {gaze_data.shape[0]}{CEND}")

            #transform_point_to_opencv(gaze_point, frame_width, frame_height)


            index_column = 1

            # Get unique indices and their first occurrence positions
            _, unique_pos = np.unique(gaze_data[:, index_column], return_index=True)

            # Select the rows corresponding to the first occurrence of each unique index
            compr_data = gaze_data[unique_pos]
            print(f"{CYELLOW}Compressed Rows: {compr_data.shape[0]} | 1st: {compr_data[0, 1]} | Last: {compr_data[-1, 1]}{CEND}")

    cap = cv2.VideoCapture(vid)

    if not cap.isOpened():
        print(f"{CRED}Error: Could not open video.{CEND}")
        exit()
    else:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if aoi else 0
        print(f"{CGREEN}Image size: {frame_width}:{frame_height} | Frames: {length} | START STREAMING - Press Q to exit{CEND}")

    num_frame = 0

    while cap.isOpened():
        if aoi:
            try:
                gaze_point = tuple(compr_data[num_frame, [3, 4]])
            except Exception as e:
                logging.error(f"Error reading gaze point: {e}")
                # print(f"Error reading gaze point: {e}")
                # exit()

        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale as the detector requires a single channel image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags in the grayscale image
        tags = detector.detect(gray)

        # Draw bounding boxes around detected AprilTags
        tags_selected = {tag.tag_id: tag for tag in tags if tag.tag_id in tag_ids} if aoi else {tag.tag_id: tag for tag in tags}

        # if len(tags_0_3) == 4:
        #     pprint(tags_0_3.keys())
        #     break

      
        for _, tag in tags_selected.items():
            if aoi:
                corner = TAGS['CORNERS'][tag.tag_id]
                if corner in (0, 1, 2, 3):
                    pt = tuple(tag.corners[TAGS['CORNERS'][tag.tag_id]].astype(int))
                else:
                    pt = (tag.center[0].astype(int), tag.corners[COORD[corner][0]][1].astype(int))
                
                cv2.circle(frame, pt, 5, (255, 255, 0), -1)

                for idx in range(len(tag.corners)):
                    
                    pt1 = tuple(tag.corners[idx].astype(int))
                    pt2 = tuple(tag.corners[(idx + 1) % len(tag.corners)].astype(int))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            # Optionally, draw the tag ID
            tag_id = str(tag.tag_id)
            cv2.putText(frame, tag_id, (tag.center[0].astype(int), tag.center[1].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        # left AOI
        ids = tag_ids[:4]
        if aoi and (len( tags_to_aoi := {tag.tag_id: tag for _, tag in tags_selected.items() if tag.tag_id in ids} ) in (3, 4)):
            res_left = draw_aoi(frame, frame_width, frame_height, tags_to_aoi, ids)
            #print(f"{num_frame} Left AOI: {res_left}", end='; ')
        # right AOI
        ids = tag_ids[2:6]
        if aoi and (len( tags_to_aoi := {tag.tag_id: tag for _, tag in tags_selected.items() if tag.tag_id in ids} ) in (3, 4)):
            res_right = draw_aoi(frame, frame_width, frame_height, tags_to_aoi, ids)
            #print(f"Right AOI: {res_right}", end='; ')
            
        #circle_mask = np.zeros_like(frame, dtype=np.uint8)
        if aoi:
            overlay = frame.copy()

            point = transform_point_to_opencv(gaze_point, frame_width, frame_height)
            #print(f"Point: {point}")
            if is_point_in_quadrilateral(point[0], point[1], res_left, num_frame):
                point_color = (0, 0, 255)
                num_gaze_left += 1
                #print(f"{CRED}Left AOI{CEND}")
            elif is_point_in_quadrilateral(point[0], point[1], res_right, num_frame):
                point_color = (0, 255, 0)
                num_gaze_right += 1
                #print(f"{CGREEN}Right AOI{CEND}")
            else:
                point_color = (50, 50, 50)
                #print(f"{CYELLOW}No AOI{CEND}")

            cv2.circle(overlay, point, 15, point_color, -1)
        
        blended_image = cv2.addWeighted(overlay, ALPHA, frame, 1-ALPHA, 0) if aoi else frame
        # Display the frame with detected tags
        cv2.imshow('Pupil Labs AprilTag Detection', blended_image)


        # Press 'q' to exit the video display loop
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        
        num_frame += 1
        # if num_frame >= 10:
        #     break
    # Release resources
    print(f"{CGREEN}Left: {num_gaze_left} | Right: {num_gaze_right}{CEND}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(aoi=AOI)



 