import ultralytics 
from ultralytics import YOLO

model = YOLO('models/best.pt')


result = model.predict('input_videos/chickenvideo_converted.mp4', save=True)

print(result[0])

for box in result[0].boxes:
    print(box)