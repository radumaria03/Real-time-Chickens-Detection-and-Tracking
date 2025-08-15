from utils import read_video, save_video
from trackers import Tracker
def main():
    # read the video
    video_frames = read_video('input_videos/chickenvideo_converted.mp4')

    # initialize tracker

    tracker = Tracker('models/best.pt')

    tracks = tracker.track(video_frames,
                           read_from_stub=True,
                           stub_path='stubs/track_stubs.pkl')
    
    # draw output 

    output_video = tracker.draw_annotations(video_frames, tracks)

    # save video
    save_video(output_video, 'output_videos/output_video.avi')


if __name__ == '__main__':
    main()