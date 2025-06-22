import cv2
import dlib
import numpy as np


SHAPE_MODEL_FILE = "models/shape_predictor_68_face_landmarks.dat"
FACE_MODEL_FILE = "models/res10_300x300_ssd_iter_140000.caffemodel"
FACE_CONFIG_FILE = "models/deploy.prototxt"


def get_lip_height(shape):
    top_lip = shape[51:53] + shape[62:64]
    bottom_lip = shape[57:59] + shape[66:68]
    top_mean = np.mean(top_lip, axis=0)
    bottom_mean = np.mean(bottom_lip, axis=0)
    return abs(top_mean[1] - bottom_mean[1])


class ActiveSpeakerDetection:
    def __init__(self, words: list, clips: list):
        self.words = words
        self.clips = clips
        self.net = cv2.dnn.readNetFromCaffe(FACE_CONFIG_FILE, FACE_MODEL_FILE)
        self.shape_predictor = dlib.shape_predictor(SHAPE_MODEL_FILE)

    def detect(self, video_path: str):
        map_words = {}
        for word in self.words:
            map_words[word['id']] = word
        cap = cv2.VideoCapture(video_path)
        for clip in self.clips:
            clip['positionX'] = self.detect_faces_in_clip(clip, map_words, cap)
        cap.release()
        # cv2.destroyAllWindows()
        return self.clips

    def detect_faces_in_clip(self, clip, map_words: dict, cap: cv2.VideoCapture):
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        face_positions = []
        for i, word_id in enumerate(clip['wordIds']):
            if i == 3:
                break
            start_frame = int(map_words[word_id]['start'] * fps)
            end_frame = int(map_words[word_id]['end'] * fps)
            # cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_range = end_frame - start_frame
            if frame_range >= 2:
                sample_frames = [start_frame, start_frame + frame_range // 2, end_frame - 1]
            elif frame_range == 1:
                sample_frames = [start_frame, end_frame]
            else:
                sample_frames = [start_frame]
            # for _ in range(start_frame, end_frame):
            # print(sample_frames)
            for frame_number in sample_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # start_time = time.time()
                faces = self.detect_faces(frame)
                # end_time = time.time()
                # detection_time = end_time - start_time
                # self.logger.info('Face detection time', {
                #     "detection_time": detection_time,
                # })
                for (x, y, w, h) in faces:
                    face_center_x = x + w / 2
                    position_x_percent = (face_center_x / frame_width) * 100
                    dlib_rect = dlib.rectangle(left=x, top=y, right=x + w, bottom=y + h)
                    shape = self.shape_predictor(gray, dlib_rect)
                    shape_np = np.array([(p.x, p.y) for p in shape.parts()])
                    lip_distance = get_lip_height(shape_np)
                    face_positions.append({
                        'position_x_percent': position_x_percent,
                        'lip_distance': lip_distance,
                    })
                    # color = (0, 255, 0)
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    # cv2.imshow('Frame', frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
        if len(face_positions) == 0:
            return 50

        def can_add_to_group(group, position):
            reference_position = group[0]['position_x_percent']
            return abs(reference_position - position['position_x_percent']) <= 10

        groups = []
        for position in face_positions:
            added_to_group = False
            for group in groups:
                if can_add_to_group(group, position):
                    group.append(position)
                    added_to_group = True
                    break
            if not added_to_group:
                groups.append([position])
        averages = []
        for group in groups:
            sum_lip_distance = sum(p['lip_distance'] for p in group)
            averages.append({
                'position_x_percent': group[0]['position_x_percent'],
                'sum_lip_distance': sum_lip_distance,
            })
        max_lip_distance_dict = max(averages, key=lambda x: x['sum_lip_distance'])
        position_x_percent = max_lip_distance_dict['position_x_percent']
        return int(position_x_percent)

    def detect_faces(self, frame, conf_threshold=0.7):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        self.net.setInput(blob)
        detections = self.net.forward()
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                faces.append([x1, y1, x2 - x1, y2 - y1])

        return faces
