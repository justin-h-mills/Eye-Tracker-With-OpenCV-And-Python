import cv2
import itertools
import numpy as np
import mediapipe as mp

def start_capture(draw=False):
    """ Captures images from webcam and annotates the images before being displayed

    Args:
        draw: Whether to draw landmarks points connections for left and right
              eye and irises
    """

    capture = cv2.VideoCapture(0)

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    while True:
        success, image = capture.read()
        if not success:
            print("Capture Not Successful")
            break

        results = get_landmarks(image, mp_face_mesh)

        if draw:
            image = draw_landmarks(image, results, mp_face_mesh, mp_drawing)

        image = cv2.flip(image, 1)

        image = eye_state(image, results, mp_face_mesh)

        cv2.imshow('Eye Tracker', image)

        user_pressed_q = cv2.waitKey(1) == ord('q')
        user_clicked_x = cv2.getWindowProperty('Eye Tracker', 0) > 0

        if user_pressed_q or user_clicked_x:
            break
    
    capture.release()

def eye_state(image, results, mp_face_mesh):
    """ Determines the state for left and right eye
    
    Args:
        image: The image to be annotated
        results: Mediapipe landmark locations for processed image
        mp_face_mesh: Mediapipe default landmark indexes

    Return:
        image: Annotated image
    """

    _, image_width, _ = image.shape

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye, right_eye = determine_state(image, face_landmarks, mp_face_mesh, 5) 
    
    image = draw_state(image, 'LEFT EYE:', left_eye, (10,30))
    image = draw_state(image, 'RIGHT EYE:', right_eye, (image_width-225,30)) 
             
    return image

def draw_state(image, eye, state, loc):
    """ Annotates image for state eye
    
    Args:
        image: The image to be annotated
        eye: The left or right eye state to be annotated
        state: State of eye of open (True) or closed (False)
        loc: The location on image to add annotation of eye state

    Return:
        image: The annotated image
    """

    if state:
        cv2.putText(image, f'{eye} OPEN', loc, cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
    else:
        cv2.putText(image, f'{eye} CLOSE', loc, cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255), 2)

    return image

def determine_state(image, results, mp_face_mesh, threshold):
    """ Determines whether left and/or right eye is open or closed
    
    Args:
        image: The image to be annotated
        results: Mediapipe landmark locations for processed image
        mp_face_mesh: Mediapipe default landmark indexes
        threshold: Threshold before a eye is considered open

    Return:
        left_eye_state: The open (True) or closed (False) state of eye
        right_eye_state: The open (True) or closed (False) state of eye
    """

    left_eye_height, _, _ = get_landmark_size(image, results, mp_face_mesh.FACEMESH_LEFT_EYE)
    right_eye_height, _, _ = get_landmark_size(image, results, mp_face_mesh.FACEMESH_RIGHT_EYE)
    face_boundary_height, _, _ = get_landmark_size(image, results, mp_face_mesh.FACEMESH_FACE_OVAL)

    left_eye_state = True if (left_eye_height/face_boundary_height)*100 > threshold else False
    right_eye_state = True if (right_eye_height/face_boundary_height)*100 > threshold else False

    return left_eye_state, right_eye_state

def get_landmark_size(image, results, landmark_indexes):
    """ Determine dimensions of facial landmarks
    
    Args:
        image: The image to be annotated
        results: Mediapipe landmark locations for processed image
        landmark_indexes: Landmark indexes for particular facial component

    Return:
        height: Height of facial component using max and min height of landmarks locations
        width: Width of facial component using max right and max left width of landmarks locations
        landmarks: Locations of all facial components landmarks
    """

    image_height, image_width, _ = image.shape
    landmark_indexes = list(itertools.chain(*landmark_indexes))
    landmarks = []

    for landmark_index in  landmark_indexes:
        landmarks.append([int(results.landmark[landmark_index].x * image_width),
                          int(results.landmark[landmark_index].y * image_height)])

    _, _, width, height = cv2.boundingRect(np.array(landmarks))

    return height, width, landmarks

def get_landmarks(image, mp_face_mesh):
    """ Processes image and returns the face landmarks locations for image
    
    Args:
        image: The image to be annotated
        mp_face_mesh: Mediapipe default landmark indexes

    Return:
        results: Mediapipe landmark locations for processed image
    """

    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(image)

    return result

def draw_landmarks(image, results, mp_face_mesh, mp_drawing):
    """ Annotates with left eye, right eye, and face boundary landmark connections
    
    Args:
        image: The image to be annotated
        results: Mediapipe landmark locations for processed image
        mp_face_mesh: Mediapipe default landmark indexes
        mp_drawing: Mediapipe image drawing tools

    Return:
        image: Annotated image
    """

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image=image,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing.DrawingSpec(thickness=2)) 
            mp_drawing.draw_landmarks(image=image,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing.DrawingSpec(thickness=2)) 
            mp_drawing.draw_landmarks(image=image,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_FACE_OVAL,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing.DrawingSpec(thickness=2))                                         
    
    return image