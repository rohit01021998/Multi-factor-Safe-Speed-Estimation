import cv2
import numpy as np
import torch
import math
import csv
import os

# Fix the current working directory issue for ultralytics
try:
    os.getcwd()
except FileNotFoundError:
    # If current directory doesn't exist, change to a valid one
    os.chdir(os.path.expanduser("~"))

from ultralytics import YOLO

class VehicleSpeedTracker:
    def __init__(self, yolo_model=None, confidence=0.2, iou=0.6):
        """
        Initialize the vehicle speed tracking system
        
        Args:
            yolo_model (str): Path to YOLO model weights
            confidence (float): Confidence threshold for detection
            iou (float): Intersection over Union threshold
        """
        if yolo_model is None:
            # Get the directory of the current script and construct the path properly
            script_dir = os.path.dirname(os.path.abspath(__file__))
            yolo_model = os.path.join(script_dir, 'ObjectDetectionRelated', 'yoloModel', 'OrientationYoloV8Model.pt')
        self.model = YOLO(yolo_model)
        self.confidence = confidence
        self.iou = iou
        
        # Tracking parameters
        self.image_width = None
        self.image_height = None
        
        # Vehicle tracking database
        self.vehicle_tracks = {}
        
        # Speed cache to avoid recalculating
        self.vehicle_speeds = {}
        
        # Safe speed cache
        self.vehicle_safe_speeds = {}
        
        # Frame-wise safe speed tracking
        self.frame_safe_speeds = {}
        
        # Vehicle size reference database
        self.vehicle_size_reference = {
            'car': {'width': 1.8, 'height': 1.5},      # Average sedan
            'truck': {'width': 2.5, 'height': 2.8},    # Large truck
            'bus': {'width': 2.6, 'height': 3.5}       # Standard bus
        }

    def estimate_distance(self, bbox, class_name='car'):
        """
        Estimate distance to vehicle using advanced heuristics
        
        Args:
            bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)
            class_name (str): Type of vehicle for more accurate estimation
        
        Returns:
            float: Estimated distance in meters
        """
        if self.image_width is None or self.image_height is None:
            # If image dimensions are not set, use default or last known dimensions
            self.image_width = 640
            self.image_height = 480
        
        # Unpack bounding box
        x1, y1, x2, y2 = bbox
        
        # Calculate bounding box dimensions
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Get reference vehicle dimensions based on class
        ref_height = self.vehicle_size_reference.get(class_name.lower(), 
                                                     self.vehicle_size_reference['car'])['height']
        
        # Perspective-based estimation
        # Add a small check to prevent division by zero
        if bbox_height > 0:
            perspective_factor = self.image_height / bbox_height
            estimated_distance = perspective_factor * ref_height
        else:
            print('################################ Distance Calculation failed ##########################################')
            estimated_distance = 10  # Default distance if calculation fails
        
        # Add some sanity checks
        estimated_distance = max(min(estimated_distance, 1000), 1)  # Limit between 5m and 1000m
        
        return estimated_distance

    def estimated_safe_speed(self, vehicle_speed_a, distance_meter):
        if distance_meter>25:
            # Convert vehicle speed from km/h to m/s
            vehicle_speed_a = (vehicle_speed_a ) * 0.27778  # Offset by 30 km/h and convert to m/s
            
            # Constants
            reaction_time = 3  # in seconds
            braking_deceleration = 3  # in m/s^2 (braking deceleration of the vehicle)
            
            try:
                # Calculate the distance covered during reaction time (distance vehicle travels before braking starts)
                distance_reaction = vehicle_speed_a * reaction_time
                
                # Calculate the remaining distance available for braking
                remaining_distance = distance_meter - distance_reaction
                
                # Check if there's enough space for braking
                if remaining_distance <= 0:
                    print("Not enough distance for safe braking.")
                    return 0
                
                # Calculate the maximum speed that can be safely achieved based on the remaining distance
                # Using the formula v^2 = 2 * a * d where 'a' is deceleration and 'd' is the remaining distance
                max_safe_speed = math.sqrt(2 * braking_deceleration * remaining_distance)
                max_safe_speed = math.sqrt(((reaction_time*braking_deceleration)**2)+2*braking_deceleration*remaining_distance) - (reaction_time*braking_deceleration)
                
                # Convert back to km/h from m/s
                safe_speed = max_safe_speed * 3.6
                
                # Debug print for checking values
                print(f"Debug - Vehicle Speed: {vehicle_speed_a * 3.6:.2f} km/h, Distance: {distance_meter:.2f}m, Safe Speed: {safe_speed:.2f} km/h")
                
                return safe_speed
            
            except Exception as e:
                print(f"Safe speed calculation error: {e}")
                return 0
        else:
            return float('inf')

    def calculate_speed(self, track_id):
        """
        Calculate vehicle speed based on distance changes
        
        Args:
            track_id (int): Unique identifier for the vehicle track
        
        Returns:
            float: Estimated speed in km/h
        """
        # Check if speed is already calculated
        if track_id in self.vehicle_speeds:
            return self.vehicle_speeds[track_id]
        
        if track_id not in self.vehicle_tracks:
            return 0
        
        track = self.vehicle_tracks[track_id]
        
        # Require at least 10 frames of tracking
        if len(track['distances']) < 20:
            return 0
        
        # Calculate distance change
        initial_distance = track['distances'][0]
        final_distance = track['distances'][-1]
        
        # Calculate speed
        distance_change = abs(final_distance - initial_distance)

        time_elapsed = len(track['distances']) / track.get('fps', 30)  # Default to 30 fps if not specified
        
        # Convert to km/h
        speed = (distance_change / time_elapsed) * 3.6  # m/s to km/h
        
        # Cache the speed
        self.vehicle_speeds[track_id] = speed
        
        print('###############################################################################')
        print(self.vehicle_speeds[track_id])

        return speed + 30

    def export_safe_speeds(self, output_csv_path):
        """
        Export frame-wise safe speeds to a CSV file
        
        Args:
            output_csv_path (str): Path to save the CSV file
        """
        with open(output_csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            
            # Write header
            header = ['Frame', 'Lowest Safe Speed', 'Vehicles Safe Speeds']
            csvwriter.writerow(header)
            
            # Sort frames to ensure chronological order
            sorted_frames = sorted(self.frame_safe_speeds.keys())
            
            for frame in sorted_frames:
                # Get safe speeds for this frame
                safe_speeds = self.frame_safe_speeds[frame]
                
                # Find lowest safe speed
                lowest_safe_speed = min(safe_speeds.values()) if safe_speeds else 'Not Detected'
                
                # Prepare vehicle safe speeds string
                vehicles_speeds_str = '; '.join([f"Track {track}: {speed:.2f} km/h" for track, speed in safe_speeds.items()])
                
                # Write row
                csvwriter.writerow([
                    frame, 
                    lowest_safe_speed, 
                    vehicles_speeds_str
                ])

    def process_video(self, video_path, output_path, safe_speeds_csv_path='safe_speeds.csv'):
        """
        Process video for vehicle detection, tracking, and speed estimation
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to output video
            safe_speeds_csv_path (str): Path to export safe speeds CSV
        """
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        self.image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (self.image_width, self.image_height))
        
        # Speed tracking results
        speed_results = []
        
        # Reset frame safe speeds tracking
        self.frame_safe_speeds = {}
        
        frame_count = 0
        last_value = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO detection and tracking
            results = self.model.track(frame, 
                                       conf=self.confidence, 
                                       iou=self.iou, 
                                       persist=True,  # Enable tracking
                                       classes=[0]  # Car, bus, truck classes in COCO
                                      )
            
            # Draw results and estimate distances
            annotated_frame = results[0].plot()
            
            # Track safe speeds for this frame
            frame_safe_speeds_dict = {}
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    bbox = (x1, y1, x2, y2)
                    
                    # Get confidence and class
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Get class name
                    class_name = self.model.names[class_id]
                    
                    # Get track ID
                    track_id = int(box.id[0]) if box.id is not None else -1
                    
                    try:
                        # Estimate distance
                        distance = self.estimate_distance(bbox, class_name)
                        
                        # Track vehicle if it has a valid track ID
                        if track_id != -1:
                            if track_id not in self.vehicle_tracks:
                                self.vehicle_tracks[track_id] = {
                                    'distances': [],
                                    'fps': fps,
                                    'class': class_name,
                                    'last_known_distance': None
                                }
                            
                            # Update track information
                            track_data = self.vehicle_tracks[track_id]
                            track_data['distances'].append(distance)
                            track_data['last_known_distance'] = distance
                            
                            # Limit to last 10 frames
                            if len(track_data['distances']) > 10:
                                track_data['distances'] = track_data['distances'][-10:]
                            
                            # Calculate speed if enough frames
                            if len(track_data['distances']) >= 10:
                                speed = self.calculate_speed(track_id)
                                
                                # Calculate safe speed
                                safe_speed = self.estimated_safe_speed(speed, track_data['last_known_distance'])
                                
                                # Store results
                                self.vehicle_safe_speeds[track_id] = safe_speed
                                frame_safe_speeds_dict[track_id] = safe_speed
                                
                                speed_results.append({
                                    'track_id': track_id,
                                    'class': class_name,
                                    'speed': speed,
                                    'safe_speed': safe_speed,
                                    'frame': frame_count
                                })
                        
                        # Display information
                        info_lines = []
                        
                        # # Add safe speed if calculated
                        # if track_id in self.vehicle_safe_speeds:
                        #     safe_speed = self.vehicle_safe_speeds[track_id]
                        #     info_lines.append(f'Safe Speed: {safe_speed:.2f}km/h')
                        
                        # # Display text
                        # for i, line in enumerate(info_lines):
                        #     cv2.putText(annotated_frame, 
                        #                 line, 
                        #                 (int(x1), int(y1 - 10 - i * 20)), 
                        #                 cv2.FONT_HERSHEY_SIMPLEX, 
                        #                 0.7, (0, 255, 0), 2)
                    
                    except Exception as e:
                        print(f"Error processing vehicle: {e}")
            
            # Store frame-wise safe speeds
            self.frame_safe_speeds[frame_count] = frame_safe_speeds_dict
            # Display lowest safe speed on top right corner
            if frame_safe_speeds_dict:
                # Optional
                # frame_safe_speeds_dict = {k: v for k, v in frame_safe_speeds_dict.items() if v != 0}
                lowest_safe_speed = min(frame_safe_speeds_dict.values())

                # lowest_safe_speed = min(float(speed) if speed != "" else float('inf') 
                #        for speed in frame_safe_speeds_dict.values())

                if (lowest_safe_speed!=0):
                    last_value = lowest_safe_speed
                    cv2.putText(annotated_frame, 
                                f'Lowest Safe Speed: {lowest_safe_speed:.2f} km/h', 
                                (self.image_width - 400, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(annotated_frame, 
                                f'Lowest Safe Speed: {last_value:.2f} km/h', 
                                (self.image_width - 400, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (0, 0, 255), 2)

            else:
                cv2.putText(annotated_frame, 
                            'No Vehicles Detected', 
                            (self.image_width - 300, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 0, 255), 2)

                # cv2.putText(annotated_frame, 
                # f'Lowest Safe Speed: {last_value:.2f} km/h', 
                # (self.image_width - 400, 30), 
                # cv2.FONT_HERSHEY_SIMPLEX, 
                # 0.7, (0, 0, 255), 2)               
            
            # Write frame to output video
            out.write(annotated_frame)
            
            resized_frame = cv2.resize(annotated_frame, (800,800))

            # Optional: Display frame
            cv2.imshow('Vehicle Speed Tracking', resized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Export safe speeds to CSV
        self.export_safe_speeds(safe_speeds_csv_path)
        
        # Print speed results
        print("Speed Tracking Results:")
        for result in speed_results:
            print(f"Track ID: {result['track_id']}, Class: {result['class']}, Speed: {result['speed']:.2f} km/h, Safe Speed: {result['safe_speed']:.2f} km/h (Frame: {result['frame']})")
        
        return speed_results

# Example usage
if __name__ == '__main__':
    tracker = VehicleSpeedTracker()
    
    # Get the parent directory and use available video files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Use the straight_2.mp4 file from the parent directory
    input_video = os.path.join(parent_dir, 'testVideos','straight_1.avi')
    output_video = os.path.join(parent_dir, 'yolo_output_video.avi')
    output_csv = os.path.join(parent_dir, 'yolo_output_video.csv')
    
    tracker.process_video(input_video, output_video, output_csv)