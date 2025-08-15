from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import sys
import torch
from torch import nn
from torchvision.models.mobilenetv3 import mobilenet_v3_small
from torchvision.transforms import v2
import torch.nn.functional as F
from PIL import Image
sys.path.append('../')
from utils import get_bbox_width



class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack(lost_track_buffer=30)
        self.mobilenetv3 = mobilenet_v3_small(pretrained=True)
        self.mobilenetv3.classifier = nn.Identity()
        self.transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ])

    def detect_frames(self, frames):
       batch_size = 20 
       detections = [] 
       for i in range(0, len(frames), batch_size):
           detection_batch = self.model.predict(frames[i : i+batch_size], conf=0.1)
           detections += detection_batch
       return detections    

    def track(self, frames, read_from_stub=False, stub_path=None):
        detections = self.detect_frames(frames)

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            print("Path found")
            with open(stub_path, 'rb')as f:
                tracks= pickle.load(f)

            return tracks


        tracks = {
            "chicken": [],
            "counted_chickens": []

        }

        counted_id = set()
        count_chicken = 0 
        check_embedding = {}
    

        for frame_num, detection in enumerate(detections):
            class_names = detection.names
            class_names_inv = {v:k for k, v in class_names.items()}

            # convert to supervision detection 

            supervision_detection = sv.Detections.from_ultralytics(detection)

            # track objects

            detection_with_tracks = self.tracker.update_with_detections(supervision_detection)

            tracks["chicken"].append({})


            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                tracker_id = frame_detection[4]

                # crop the image using the bounding box points 
                x1, y1, x2, y2 = map(int, bbox)
                frame = frames[frame_num]
                crop = frame[y1:y2, x1:x2]
                crop_img = Image.fromarray(crop)
                crop_img_tensor = self.transform(crop_img).unsqueeze(dim=0)
                self.mobilenetv3.eval()
                with torch.no_grad():
                  embedding = self.mobilenetv3(crop_img_tensor).squeeze().cpu().numpy()
  

                tracks["chicken"][frame_num][tracker_id] = {"bbox": bbox,
                                                            "embedding": embedding}
                

                similarity_threshold = 0.5

                if cls_id == class_names_inv['chicken']:
                    duplicate = False

                    # check the similarity of the embeddings
                    for stored_embedding in check_embedding.values():
                        similarity = F.cosine_similarity(torch.tensor(stored_embedding).unsqueeze(0), 
                                                         torch.tensor(embedding).unsqueeze(0)).item()
                        if similarity > similarity_threshold:
                            duplicate = True
                            break

                
                    if not duplicate:
                        counted_id.add(tracker_id)
                        check_embedding[tracker_id] = embedding
                        count_chicken +=  1
                        

            tracks['counted_chickens'].append((count_chicken))

        if stub_path is not None:
                print("No path found")
                with open(stub_path, "wb") as f:
                    pickle.dump(tracks,f)

        return tracks
    
    def draw_rectangle(self, frame, bbox, color, track_id, text):

        x1, y1, x2, y2 = map(int, bbox)
        start_point = (x1, y1)
        end_point = (x2, y2)

        cv2.rectangle(
            frame,
            start_point,
            end_point,
            color=color,
            thickness=3
        )

        font_name = cv2.FONT_HERSHEY_PLAIN
        wcolor = (0, 0, 0)

        origine =  (x1, y1)

       

        cv2.putText(
            frame,
            text=text,
            org=origine,
            fontFace=font_name,
            fontScale=12,
            color=wcolor,
            thickness=6
        )
        return frame
    

    def draw_annotations(self, video_frames, tracks):
        output_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()


            chickens_dict = tracks['chicken'][frame_num]
            counter = tracks['counted_chickens'][frame_num]

            # draw ckicken 

            for track_id, chicken in chickens_dict.items():
                frame  = self.draw_rectangle(frame, chicken["bbox"], (225, 0, 0), track_id, str(counter))

        

            output_frames.append(frame)

        return output_frames


