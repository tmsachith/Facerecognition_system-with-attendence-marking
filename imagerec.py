import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta
import pandas as pd
import threading
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import base64
import time
import json

app = Flask(__name__, static_folder='static')

# Function to find encodings for known faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except IndexError:
            print(f"No face found in one of the images. Skipping...")
    return encodeList

# Function to mark attendance in CSV file
def markAttendance(name):
    now = datetime.now()
    dateString = now.strftime('%Y-%m-%d')
    timeString = now.strftime('%H:%M:%S')
    
    # Create attendance file if it doesn't exist
    filename = f'Attendance_{dateString}.csv'
    if not os.path.isfile(filename):
        df = pd.DataFrame(columns=['Name', 'Time', 'Date'])
        df.to_csv(filename, index=False)
    
    # Read existing attendance
    df = pd.read_csv(filename)
    
    # Check if person is already marked for today
    if not df['Name'].str.contains(name).any():
        new_row = {'Name': name, 'Time': timeString, 'Date': dateString}
        df = df._append(new_row, ignore_index=True)
        df.to_csv(filename, index=False)
        return f"{name} - Marked"
    else:
        return f"{name} - Already Marked"

# Global variables for thread-safe operation
frame_lock = threading.Lock()
current_frame = None
registration_frame = None
recognition_events = []
classNames = []
encodeListKnown = []
camera_running = False
registration_camera_running = False

# Load initial images and encodings
def initialize_face_recognition():
    global classNames, encodeListKnown
    
    # Path to your images folder
    path = 'ImagesAttendance'
    
    # Create directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
        return [], []  # Return empty lists since there are no images yet
    
    images = []
    classNames = []
    myList = os.listdir(path)
    
    if not myList:
        print("No images found in the ImagesAttendance directory")
        return [], []
    
    # Load all images and names
    for cl in myList:
        if cl.lower().endswith(('.png', '.jpg', '.jpeg')):
            curImg = cv2.imread(f'{path}/{cl}')
            if curImg is not None:
                images.append(curImg)
                classNames.append(os.path.splitext(cl)[0])  # Remove file extension
            else:
                print(f"Could not read image: {cl}")
    
    print('Encoding Images...')
    encodeListKnown = findEncodings(images)
    print('Encoding Complete')
    
    return classNames, encodeListKnown

# Initialize face recognition
classNames, encodeListKnown = initialize_face_recognition()

# Function to generate frames for streaming
def generate_frames():
    global current_frame, camera_running
    
    while True:
        with frame_lock:
            if current_frame is not None and camera_running:
                ret, buffer = cv2.imencode('.jpg', current_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)  # Limit to roughly 30 FPS

# Function to generate frames for registration camera
def generate_registration_frames():
    global registration_frame, registration_camera_running
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam for registration")
        return
    
    while True:
        if registration_camera_running:
            success, img = cap.read()
            if not success:
                break
            
            # Add a frame to indicate this is for capture
            cv2.putText(img, "Click 'Capture' to take photo", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)
            cv2.rectangle(img, (100, 100), (540, 380), (0, 255, 0), 2)  # Frame for face positioning
            
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            
            # Update the global frame
            with frame_lock:
                registration_frame = img
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.1)  # Sleep when camera is not active
    
    cap.release()

# Function to count students registered
def count_students():
    path = 'ImagesAttendance'
    if not os.path.exists(path):
        return 0
    
    return len([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Function to count students present today
def count_present_today():
    now = datetime.now()
    dateString = now.strftime('%Y-%m-%d')
    filename = f'Attendance_{dateString}.csv'
    
    if not os.path.isfile(filename):
        return 0
    
    df = pd.read_csv(filename)
    return len(df)

# Get attendance data for a specific date
def get_attendance_for_date(date_str):
    filename = f'Attendance_{date_str}.csv'
    
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        attendance_data = []
        
        for _, row in df.iterrows():
            attendance_data.append({
                'name': row['Name'],
                'time': row['Time'],
                'status': 'Present'
            })
            
        return attendance_data
    else:
        return []

# Get attendance summary report for a date range
def get_attendance_report(start_date, end_date):
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Dictionary to store each student's attendance
    report = {}
    
    # Dictionary to keep track of all dates in the range
    all_dates = {}
    
    # Generate all dates in the range
    current_date = start
    while current_date <= end:
        date_str = current_date.strftime('%Y-%m-%d')
        all_dates[date_str] = True
        current_date += timedelta(days=1)
    
    # Process each date in the range
    current_date = start
    while current_date <= end:
        date_str = current_date.strftime('%Y-%m-%d')
        filename = f'Attendance_{date_str}.csv'
        
        if os.path.isfile(filename):
            df = pd.read_csv(filename)
            
            for _, row in df.iterrows():
                name = row['Name']
                
                if name not in report:
                    report[name] = {
                        'name': name,
                        'totalPresent': 0,
                        'totalDays': len(all_dates),
                        'percentage': 0,
                        'dates': {}
                    }
                
                # Mark this date as present
                report[name]['dates'][date_str] = 'Present'
                report[name]['totalPresent'] += 1
        
        current_date += timedelta(days=1)
    
    # Calculate attendance percentage for each student
    for student in report.values():
        student['percentage'] = round((student['totalPresent'] / student['totalDays']) * 100, 1)
    
    return list(report.values())

# Routes for the web application
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/registration_feed')
def registration_feed():
    return Response(generate_registration_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/events')
def get_events():
    global recognition_events
    return jsonify(recognition_events[-10:])

@app.route('/api/stats')
def get_stats():
    total_students = count_students()
    present_today = count_present_today()
    now = datetime.now()
    
    return jsonify({
        'totalStudents': total_students,
        'presentToday': present_today,
        'date': now.strftime('%Y-%m-%d'),
        'time': now.strftime('%H:%M:%S')
    })

@app.route('/api/attendance')
def get_attendance():
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    return jsonify(get_attendance_for_date(date))

@app.route('/api/report', methods=['GET'])
def get_report():
    start_date = request.args.get('start_date', (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))
    end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    
    report_data = get_attendance_report(start_date, end_date)
    return jsonify(report_data)

@app.route('/api/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera_running
    
    data = request.json
    camera_running = data.get('status', False)
    
    return jsonify({'success': True, 'camera_running': camera_running})

@app.route('/api/toggle_registration_camera', methods=['POST'])
def toggle_registration_camera():
    global registration_camera_running, camera_running
    
    data = request.json
    registration_camera_running = data.get('status', False)
    
    # Turn off main camera if registration camera is turned on
    if registration_camera_running:
        camera_running = False
    
    return jsonify({
        'success': True, 
        'registration_camera_running': registration_camera_running,
        'camera_running': camera_running
    })

@app.route('/api/capture_registration_image', methods=['GET'])
def capture_registration_image():
    global registration_frame
    
    with frame_lock:
        if registration_frame is not None:
            ret, buffer = cv2.imencode('.jpg', registration_frame)
            captured_image = base64.b64encode(buffer).decode('utf-8')
            return jsonify({'success': True, 'image': captured_image})
        else:
            return jsonify({'success': False, 'message': 'No camera frame available'})

@app.route('/register', methods=['POST'])
def register_student():
    global classNames, encodeListKnown
    
    # Handle the photo from capture or file upload
    name = request.form.get('name')
    
    if not name:
        return jsonify({'success': False, 'message': 'No name provided'})
    
    # Path to save images
    path = 'ImagesAttendance'
    if not os.path.exists(path):
        os.makedirs(path)
    
    filepath = os.path.join(path, f'{name}.jpg')
    
    if 'photo' in request.files:
        # Handle file upload
        file = request.files['photo']
        if file.filename != '':
            file.save(filepath)
    elif 'captured_photo' in request.form:
        # Handle captured photo from webcam
        encoded_image = request.form.get('captured_photo')
        if encoded_image.startswith('data:image/'):
            encoded_image = encoded_image.split(',')[1]
        
        # Decode and save image
        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(encoded_image))
    else:
        return jsonify({'success': False, 'message': 'No photo provided'})
    
    # Load the new image and add to encodings
    new_img = cv2.imread(filepath)
    if new_img is not None:
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
        try:
            new_encode = face_recognition.face_encodings(new_img)[0]
            
            with frame_lock:  # Lock to safely modify shared resources
                encodeListKnown.append(new_encode)
                classNames.append(name)
                
            return jsonify({'success': True, 'message': 'Student registered successfully'})
        except IndexError:
            # Remove the file if no face is detected
            os.remove(filepath)
            return jsonify({'success': False, 'message': 'No face detected in the uploaded image'})
    else:
        return jsonify({'success': False, 'message': 'Failed to process the uploaded image'})

@app.route('/templates/<path:path>')
def serve_template(path):
    return send_from_directory('templates', path)

# Face recognition thread
def face_recognition_thread():
    global current_frame, recognition_events, classNames, encodeListKnown, camera_running
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting face recognition thread")
    
    while True:
        if camera_running and not registration_camera_running:
            success, img = cap.read()
            if not success:
                print("Failed to get frame from webcam")
                time.sleep(1)  # Wait before retrying
                cap = cv2.VideoCapture(0)  # Try to reconnect
                continue
            
            # Make a copy of the image for processing
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Resize for faster processing
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            
            # Check if there are any encodings
            if len(encodeListKnown) == 0:
                # Just display the frame with a message if no faces are registered
                cv2.putText(img, "No faces registered", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                with frame_lock:
                    current_frame = img
                time.sleep(0.03)
                continue
            
            # Find all faces in current frame
            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
            
            # Check each face against known encodings
            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                with frame_lock:  # Lock when accessing shared resources
                    local_encodeListKnown = encodeListKnown.copy()
                    local_classNames = classNames.copy()
                
                matches = face_recognition.compare_faces(local_encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(local_encodeListKnown, encodeFace)
                
                if len(faceDis) > 0:  # Check if the list is not empty
                    matchIndex = np.argmin(faceDis)
                    
                    # If match found, mark attendance
                    if matches[matchIndex]:
                        name = local_classNames[matchIndex].upper()
                        attendance_status = markAttendance(name)
                        
                        # Add to recognition events
                        now = datetime.now()
                        event = {
                            'name': name,
                            'status': 'Marked' if 'Marked' in attendance_status else 'Already Marked',
                            'time': now.strftime('%H:%M:%S')
                        }
                        recognition_events.insert(0, event)
                        if len(recognition_events) > 20:
                            recognition_events.pop()
                        
                        # Draw rectangle and name
                        y1, x2, y2, x1 = faceLoc
                        # Scale back up face locations since we scaled down the image
                        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, attendance_status, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        # Unknown face
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 0, 255), cv2.FILLED)
                        cv2.putText(img, "Unknown", (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            
            # Update the current frame
            with frame_lock:
                current_frame = img
        else:
            # Release camera when not in use
            if cap.isOpened():
                time.sleep(0.1)  # Sleep briefly to reduce CPU usage
            else:
                cap = cv2.VideoCapture(0)
        
        # Sleep to reduce CPU usage
        time.sleep(0.03)
    
    cap.release()

# Create the templates directory and save the HTML
def setup_templates():
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    
    # Save the HTML to templates/index.html
    with open(os.path.join(templates_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Attendance System</title>
    <style>
        :root {
            --primary: #4361ee;
            --secondary: #3f37c9;
            --success: #4cc9f0;
            --light: #f8f9fa;
            --dark: #212529;
            --accent: #f72585;
            --text: #495057;
            --border-radius: 12px;
            --shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background-color: #f5f7ff;
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            width: 100%;
        }

        header {
            background-color: white;
            box-shadow: var(--shadow);
            padding: 15px 0;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 24px;
            font-weight: 700;
            color: var(--primary);
        }

        .logo-icon {
            width: 40px;
            height: 40px;
            background-color: var(--primary);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
        }

        .tabs {
            display: flex;
            margin-top: 30px;
            border-bottom: 1px solid #e9ecef;
            gap: 10px;
        }

        .tab {
            padding: 12px 20px;
            cursor: pointer;
            font-weight: 600;
            color: var(--text);
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
        }

        .tab.active {
            color: var(--primary);
            border-bottom: 3px solid var(--primary);
        }

        .tab:hover {
            color: var(--primary);
        }

        .content {
            padding: 30px 0;
            display: none;
        }

        .content.active {
            display: block;
        }

        .card {
            background-color: white;
            border-radius: var(--border-radius);
            box-shadow: var(--shadow);
            overflow: hidden;
            margin-bottom: 20px;
        }

        .card-header {
            padding: 20px;
            border-bottom: 1px solid #e9ecef;
            font-weight: 600;
            font-size: 18px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .card-body {
            padding: 20px;
        }

        .camera-container {
            position: relative;
            height: 480px;
            overflow: hidden;
            border-radius: var(--border-radius);
            background-color: #eee;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        #camera-feed, #registration-camera-feed {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .camera-overlay {
            position: absolute;
            top: 10px;
            right: 10px;
            background-color: rgba(0, 0, 0, 0.6);
            color: white;
            padding: 8px 12px;
            border-radius: 20px;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .camera-status {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background-color: #4ade80;
        }

        .camera-controls {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
        }

        .btn {
            padding: 12px 24px;
            border-radius: 6px;
            border: none;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .btn-primary {
            background-color: var(--primary);
            color: white;
        }

        .btn-primary:hover {
            background-color: var(--secondary);
        }

        .btn-secondary {
            background-color: #f1f3f5;
            color: var(--text);
        }

        .btn-secondary:hover {
            background-color: #e9ecef;
        }

        .btn-danger {
            background-color: var(--accent);
            color: white;
        }

        .btn-danger:hover {
            background-color: #d61a6c;
        }

        .attendance-list {
            display: grid;
            gap: 15px;
        }

        .attendance-item {
            display: flex;
            align-items: center;
            padding: 15px;
            background-color: white;
            border-radius: var(--border-radius);
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s ease;
        }

        .attendance-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
        }

        .attendance-avatar {
            width: 45px;
            height: 45px;
            border-radius: 50%;
            margin-right: 15px;
            overflow: hidden;
            background-color: #e9ecef;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            color: var(--primary);
        }

        .attendance-avatar img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .attendance-details {
            flex: 1;
        }

        .attendance-name {
            font-weight: 600;
            margin-bottom: 3px;
        }

        .attendance-time {
            font-size: 14px;
            color: #6c757d;
        }

        .attendance-status {
            font-size: 14px;
            padding: 4px 10px;
            border-radius: 20px;
            background-color: #e2f3ff;
            color: #0891b2;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-card {
            background-color: white;
            border-radius: var(--border-radius);
            padding: 20px;
            box-shadow: var(--shadow);
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .stat-icon {
            width: 50px;
            height: 50px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            color: white;
        }

        .stat-icon.blue {
            background-color: var(--primary);
        }

        .stat-icon.purple {
            background-color: var(--secondary);
        }

        .stat-icon.teal {
            background-color: var(--success);
        }

        .stat-icon.pink {
            background-color: var(--accent);
        }

        .stat-details h3 {
            font-size: 24px;
            margin-bottom: 5px;
        }

        .stat-details p {
            color: #6c757d;
            font-size: 14px;
        }

        .registration-form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
        }

        input[type="text"],
        input[type="email"],
        input[type="date"] {
            width: 100%;
            padding: 12px 15px;
            border: 1px solid #ced4da;
            border-radius: 6px;
            font-size: 16px;
            transition: border-color 0.3s ease;
        }

        input[type="text"]:focus,
        input[type="email"]:focus,
        input[type="date"]:focus {
            border-color: var(--primary);
            outline: none;
        }

        .photo-upload {
            border: 2px dashed #ced4da;
            border-radius: var(--border-radius);
            padding: 30px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .photo-upload:hover {
            border-color: var(--primary);
        }

        .photo-upload-icon {
            font-size: 40px;
            color: #adb5bd;
            margin-bottom: 15px;
        }

        .preview-container {
            margin-top: 20px;
            display: none;
        }

        #photo-preview {
            max-width: 100%;
            max-height: 200px;
            border-radius: 8px;
        }

        .form-actions {
            margin-top: 20px;
            display: flex;
            gap: 10px;
        }

        .recent-recognitions {
            margin-top: 20px;
        }

        .recognition-log {
            max-height: 200px;
            overflow-y: auto;
            background-color: #f8f9fa;
            border-radius: 6px;
            padding: 10px;
            font-family: monospace;
            font-size: 14px;
        }

        .recognition-entry {
            padding: 5px 0;
            border-bottom: 1px solid #e9ecef;
        }

        .recognition-entry:last-child {
            border-bottom: none;
        }

        .recognition-time {
            color: #6c757d;
            font-size: 12px;
            margin-right: 10px;
        }

        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 5px;
        }

        .status-success {
            background-color: #4ade80;
        }

        .status-pending {
            background-color: #f59e0b;
        }

        .toggle-switch {
            position: relative;
            display: inline-block;
            width: 60px;
            height: 34px;
            margin-left: 10px;
        }
        
        .toggle-switch input { 
            opacity: 0;
            width: 0;
            height: 0;
        }
        
        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 34px;
        }
        
        .slider:before {
            position: absolute;
            content: "";
            height: 26px;
            width: 26px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }
        
        input:checked + .slider {
            background-color: var(--primary);
        }
        
        input:focus + .slider {
            box-shadow: 0 0 1px var(--primary);
        }
        
        input:checked + .slider:before {
            transform: translateX(26px);
        }

        @media (max-width: 768px) {
            .tabs {
                overflow-x: auto;
                white-space: nowrap;
                padding-bottom: 5px;
            }
            
            .camera-container {
                height: 300px;
            }
            
            .btn {
                padding: 10px 16px;
                font-size: 14px;
            }
            
            .registration-form {
                grid-template-columns: 1fr;
            }
        }

        .connection-status {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
        }

        .connection-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }

        .connected {
            background-color: #4ade80;
        }

        .disconnected {
            background-color: #ef4444;
        }

        .connecting {
            background-color: #f59e0b;
            animation: pulse 1.5s infinite;
        }

        /* New styles for report section */
        .report-controls {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 20px;
        }

        .date-range {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .report-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }

        .report-table th, .report-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }

        .report-table th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: var(--text);
        }

        .report-table tr:hover {
            background-color: #f8f9fa;
        }

        .attendance-percentage {
            background-color: #e2f3ff;
            padding: 4px 10px;
            border-radius: 20px;
            font-weight: 600;
        }

        .high {
            background-color: #dcfce7;
            color: #16a34a;
        }

        .medium {
            background-color: #fef9c3;
            color: #ca8a04;
        }

        .low {
            background-color: #fee2e2;
            color: #dc2626;
        }

        .date-picker-wrapper {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .hidden {
            display: none;
        }

        .tabs-content-container {
            min-height: 300px;
        }

        .photo-method-toggle {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }

        .photo-method {
            padding: 10px 15px;
            background: #f8f9fa;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s;
        }

        .photo-method.active {
            background: var(--primary);
            color: white;
        }

        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <div class="navbar">
                <div class="logo">
                    <div class="logo-icon">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 4c1.93 0 3.5 1.57 3.5 3.5S13.93 13 12 13s-3.5-1.57-3.5-3.5S10.07 6 12 6zm0 14c-2.03 0-4.43-.82-6.14-2.88C7.55 15.8 9.68 15 12 15s4.45.8 6.14 2.12C16.43 19.18 14.03 20 12 20z"/>
                        </svg>
                    </div>
                    <span>Smart Attendance</span>
                </div>
            </div>
        </div>
    </header>

    <div class="container">
        <div class="tabs">
            <div class="tab active" data-tab="dashboard">Dashboard</div>
            <div class="tab" data-tab="attendance">Attendance</div>
            <div class="tab" data-tab="register">Register</div>
            <div class="tab" data-tab="reports">Reports</div>
        </div>

        <div class="tabs-content-container">
            <div class="content active" id="dashboard">
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-icon blue">
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                                <circle cx="9" cy="7" r="4"></circle>
                                <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
                                <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
                            </svg>
                        </div>
                        <div class="stat-details">
                            <h3 id="total-students">0</h3>
                            <p>Total Students</p>
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon purple">
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"></path>
                                <circle cx="9" cy="7" r="4"></circle>
                                <path d="M22 21v-2a4 4 0 0 0-3-3.87"></path>
                                <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
                            </svg>
                        </div>
                        <div class="stat-details">
                            <h3 id="present-today">0</h3>
                            <p>Present Today</p>
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon teal">
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
                                <line x1="16" y1="2" x2="16" y2="6"></line>
                                <line x1="8" y1="2" x2="8" y2="6"></line>
                                <line x1="3" y1="10" x2="21" y2="10"></line>
                            </svg>
                        </div>
                        <div class="stat-details">
                            <h3 id="current-date">-</h3>
                            <p>Current Date</p>
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon pink">
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <circle cx="12" cy="12" r="10"></circle>
                                <polyline points="12 6 12 12 16 14"></polyline>
                            </svg>
                        </div>
                        <div class="stat-details">
                            <h3 id="current-time">-</h3>
                            <p>Current Time</p>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <span>Live Attendance</span>
                        <div style="display: flex; align-items: center;">
                            <div class="connection-status">
                                <div id="connection-indicator" class="connection-indicator disconnected"></div>
                                <span id="connection-text">Disconnected</span>
                            </div>
                            <div style="display: flex; align-items: center; margin-left: 15px;">
                                <span>Camera:</span>
                                <label class="toggle-switch">
                                    <input type="checkbox" id="camera-toggle">
                                    <span class="slider"></span>
                                </label>
                            </div>
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="camera-container">
                            <img id="camera-feed" alt="Camera feed" src="/video_feed">
                            <div id="camera-off-message" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; display: none;">
                                <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#adb5bd" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M16 16v1a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h2m5.66 0H14a2 2 0 0 1 2 2v3.34l1 1L23 7v10"></path>
                                    <line x1="1" y1="1" x2="23" y2="23"></line>
                                </svg>
                                <p style="margin-top: 10px; color: #6c757d;">Camera is turned off</p>
                                <button id="start-camera" class="btn btn-primary" style="margin-top: 15px;">
                                    Start Camera
                                </button>
                            </div>
                        </div>
                        <div class="recent-recognitions">
                            <h4>Recent Recognitions</h4>
                            <div class="recognition-log" id="recognition-log">
                                <div class="recognition-entry">Waiting for recognition events...</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="content" id="attendance">
                <div class="card">
                    <div class="card-header">
                        <span>Attendance Records</span>
                        <div class="date-picker-wrapper">
                            <label for="attendance-date">Date:</label>
                            <input type="date" id="attendance-date" class="form-control">
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="attendance-list" id="attendance-list">
                            <!-- Attendance list will be populated dynamically -->
                            <div class="recognition-entry">No attendance records yet for today.</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="content" id="register">
                <div class="card">
                    <div class="card-header">
                        Register New Student
                    </div>
                    <div class="card-body">
                        <div class="photo-method-toggle">
                            <div class="photo-method active" data-method="upload" id="upload-method">
                                Upload Photo
                            </div>
                            <div class="photo-method" data-method="capture" id="capture-method">
                                Capture from Camera
                            </div>
                        </div>
                        
                        <form id="registration-form" class="registration-form">
                            <div>
                                <div class="form-group">
                                    <label for="student-name">Student Name</label>
                                    <input type="text" id="student-name" placeholder="Enter full name" required>
                                </div>
                                <div class="form-group">
                                    <label for="student-id">Student ID</label>
                                    <input type="text" id="student-id" placeholder="Enter student ID" required>
                                </div>
                                <div class="form-group">
                                    <label for="student-email">Email</label>
                                    <input type="email" id="student-email" placeholder="Enter email address">
                                </div>
                            </div>
                            <div>
                                <!-- Upload Photo Method -->
                                <div id="upload-photo-container">
                                    <div class="form-group">
                                        <label>Student Photo</label>
                                        <div class="photo-upload" id="photo-upload">
                                            <div class="photo-upload-icon">
                                                <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                                    <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7"></path>
                                                    <line x1="16" y1="5" x2="22" y2="5"></line>
                                                    <line x1="19" y1="2" x2="19" y2="8"></line>
                                                    <circle cx="9" cy="9" r="2"></circle>
                                                    <path d="M21 15l-3.086-3.086a2 2 0 0 0-2.828 0L6 21"></path>
                                                </svg>
                                            </div>
                                            <p>Click to upload photo</p>
                                            <input type="file" id="photo-input" accept="image/*" style="display: none;">
                                        </div>
                                        <div class="preview-container" id="preview-container">
                                            <img id="photo-preview" src="" alt="Photo preview">
                                        </div>
                                    </div>
                                </div>
                                
                                <!-- Capture Photo Method -->
                                <div id="capture-photo-container" style="display: none;">
                                    <div class="form-group">
                                        <label>Capture Photo</label>
                                        <div class="camera-container" style="height: 300px;">
                                            <img id="registration-camera-feed" alt="Camera feed" src="/registration_feed">
                                        </div>
                                        <div class="camera-controls">
                                            <button type="button" class="btn btn-primary" id="capture-btn">
                                                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                                    <circle cx="12" cy="12" r="10"></circle>
                                                    <circle cx="12" cy="12" r="3"></circle>
                                                </svg>
                                                Capture
                                            </button>
                                        </div>
                                        <div class="preview-container" id="capture-preview-container">
                                            <img id="capture-preview" src="" alt="Captured photo preview">
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="form-actions">
                                    <button type="submit" class="btn btn-primary">Register Student</button>
                                    <button type="button" class="btn btn-secondary" id="reset-form">Reset Form</button>
                                </div>
                            </div>
                        </form>
                    </div>
                </div>
            </div>

            <div class="content" id="reports">
                <div class="card">
                    <div class="card-header">
                        <span>Attendance Reports</span>
                    </div>
                    <div class="card-body">
                        <div class="report-controls">
                            <div class="date-range">
                                <label for="start-date">From:</label>
                                <input type="date" id="start-date" class="form-control">
                                
                                <label for="end-date">To:</label>
                                <input type="date" id="end-date" class="form-control">
                                
                                <button id="generate-report" class="btn btn-primary">Generate Report</button>
                            </div>
                        </div>
                        
                        <div id="report-content">
                            <!-- Report will be generated here -->
                            <p>Select a date range and click "Generate Report" to see attendance statistics.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Tab switching functionality
        const tabs = document.querySelectorAll('.tab');
        const contents = document.querySelectorAll('.content');

        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const target = tab.dataset.tab;
                
                // Remove active class from all tabs and contents
                tabs.forEach(t => t.classList.remove('active'));
                contents.forEach(c => c.classList.remove('active'));
                
                // Add active class to clicked tab and corresponding content
                tab.classList.add('active');
                document.getElementById(target).classList.add('active');
                
                // Special handling for registration tab - turn off main camera if going to registration
                if (target === 'register') {
                    toggleMainCamera(false);
                    if (document.querySelector('.photo-method[data-method="capture"]').classList.contains('active')) {
                        toggleRegistrationCamera(true);
                    }
                } else {
                    toggleRegistrationCamera(false);
                }
            });
        });

        // Set current date and time
        function updateDateTime() {
            const now = new Date();
            const dateOptions = { year: 'numeric', month: 'short', day: 'numeric' };
            const timeOptions = { hour: '2-digit', minute: '2-digit' };
            
            document.getElementById('current-date').textContent = now.toLocaleDateString('en-US', dateOptions);
            document.getElementById('current-time').textContent = now.toLocaleTimeString('en-US', timeOptions);
        }
        
        updateDateTime();
        setInterval(updateDateTime, 60000); // Update every minute

        // Toggle main camera
        const cameraToggle = document.getElementById('camera-toggle');
        const cameraFeed = document.getElementById('camera-feed');
        const cameraOffMessage = document.getElementById('camera-off-message');
        const startCameraBtn = document.getElementById('start-camera');
        
        cameraToggle.addEventListener('change', function() {
            toggleMainCamera(this.checked);
        });
        
        startCameraBtn.addEventListener('click', function() {
            cameraToggle.checked = true;
            toggleMainCamera(true);
        });
        
        function toggleMainCamera(status) {
            fetch('/api/toggle_camera', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ status: status }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.camera_running) {
                    cameraFeed.style.display = 'block';
                    cameraOffMessage.style.display = 'none';
                    // Make sure toggle switch reflects server state
                    cameraToggle.checked = true;
                } else {
                    cameraFeed.style.display = 'none';
                    cameraOffMessage.style.display = 'flex';
                    cameraToggle.checked = false;
                }
            })
            .catch(error => {
                console.error('Error toggling camera:', error);
            });
        }
        
        // Toggle registration camera
        function toggleRegistrationCamera(status) {
            fetch('/api/toggle_registration_camera', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ status: status }),
            })
            .then(response => response.json())
            .catch(error => {
                console.error('Error toggling registration camera:', error);
            });
        }

        // Photo upload and preview
        const photoUpload = document.getElementById('photo-upload');
        const photoInput = document.getElementById('photo-input');
        const photoPreview = document.getElementById('photo-preview');
        const previewContainer = document.getElementById('preview-container');
        const resetFormBtn = document.getElementById('reset-form');
        
        // Registration photo method switching
        const uploadMethod = document.getElementById('upload-method');
        const captureMethod = document.getElementById('capture-method');
        const uploadPhotoContainer = document.getElementById('upload-photo-container');
        const capturePhotoContainer = document.getElementById('capture-photo-container');
        
        uploadMethod.addEventListener('click', function() {
            uploadMethod.classList.add('active');
            captureMethod.classList.remove('active');
            uploadPhotoContainer.style.display = 'block';
            capturePhotoContainer.style.display = 'none';
            toggleRegistrationCamera(false);
        });
        
        captureMethod.addEventListener('click', function() {
            captureMethod.classList.add('active');
            uploadMethod.classList.remove('active');
            capturePhotoContainer.style.display = 'block';
            uploadPhotoContainer.style.display = 'none';
            toggleRegistrationCamera(true);
        });

        photoUpload.addEventListener('click', () => {
            photoInput.click();
        });

        photoInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    photoPreview.src = event.target.result;
                    previewContainer.style.display = 'block';
                };
                reader.readAsDataURL(e.target.files[0]);
            }
        });
        
        // Capture photo from webcam
        const captureBtn = document.getElementById('capture-btn');
        const capturePreview = document.getElementById('capture-preview');
        const capturePreviewContainer = document.getElementById('capture-preview-container');
        
        captureBtn.addEventListener('click', function() {
            fetch('/api/capture_registration_image')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        capturePreview.src = `data:image/jpeg;base64,${data.image}`;
                        capturePreviewContainer.style.display = 'block';
                    } else {
                        alert('Failed to capture image: ' + data.message);
                    }
                })
                .catch(error => {
                    console.error('Error capturing image:', error);
                });
        });

        resetFormBtn.addEventListener('click', () => {
            document.getElementById('registration-form').reset();
            previewContainer.style.display = 'none';
            capturePreviewContainer.style.display = 'none';
        });

        // Registration form submission
        const registrationForm = document.getElementById('registration-form');
        
        registrationForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const name = document.getElementById('student-name').value;
            const id = document.getElementById('student-id').value;
            const email = document.getElementById('student-email').value;
            
            // Create form data to send to backend
            const formData = new FormData();
            formData.append('name', `${name}-${id}`); // Include ID in the name for uniqueness
            
            // Check which photo method was used
            const isCapture = document.querySelector('.photo-method[data-method="capture"]').classList.contains('active');
            
            if (isCapture) {
                // Using webcam capture
                if (!capturePreview.src || capturePreview.src === '') {
                    alert('Please capture a photo first');
                    return;
                }
                
                // Convert data URL to base64 string
                let base64String = capturePreview.src;
                if (base64String.includes('base64,')) {
                    base64String = base64String.split('base64,')[1];
                }
                
                formData.append('captured_photo', base64String);
            } else {
                // Using file upload
                const fileInput = document.getElementById('photo-input');
                if (!fileInput.files || fileInput.files.length === 0) {
                    alert('Please upload a photo of the student');
                    return;
                }
                
                formData.append('photo', fileInput.files[0]);
            }
            
            try {
                // Send data to backend
                const response = await fetch('/register', {
                   method: 'POST',
                   body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                   alert('Student registered successfully!');
                   registrationForm.reset();
                   previewContainer.style.display = 'none';
                   capturePreviewContainer.style.display = 'none';
                   updateStats(); // Update the stats after successful registration
                } else {
                   alert('Registration failed: ' + result.message);
                }
                
            } catch (error) {
                console.error('Error submitting form:', error);
                alert('Registration failed. Please try again.');
            }
        });

        // Handle connection to Python backend
        const connectionIndicator = document.getElementById('connection-indicator');
        const connectionText = document.getElementById('connection-text');
        const recognitionLog = document.getElementById('recognition-log');
        
        // Function to add recognition events to the log
        function addRecognitionEvent(name, status, time) {
            const statusClass = status.includes('Already') ? 'status-pending' : 'status-success';
            
            const entry = document.createElement('div');
            entry.className = 'recognition-entry';
            entry.innerHTML = `
                <span class="recognition-time">${time}</span>
                <span class="status-indicator ${statusClass}"></span>
                <span>${name}: ${status}</span>
            `;
            
            // Clear initial "waiting" message if it exists
            if (recognitionLog.children.length === 1 && 
                recognitionLog.children[0].textContent.includes('Waiting')) {
                recognitionLog.innerHTML = '';
            }
            
            recognitionLog.insertBefore(entry, recognitionLog.firstChild);
            
            // Limit number of entries
            if (recognitionLog.children.length > 20) {
                recognitionLog.removeChild(recognitionLog.lastChild);
            }
        }

        // Function to fetch and update recognition events
        async function fetchEvents() {
            try {
                const response = await fetch('/api/events');
                const events = await response.json();
                
                // Clear current log if we have events
                if (events.length > 0) {
                    recognitionLog.innerHTML = '';
                    
                    // Add events to log in reverse order (newest first)
                    events.forEach(event => {
                        addRecognitionEvent(event.name, event.status, event.time);
                    });
                }
            } catch (error) {
                console.error('Error fetching events:', error);
                // If there's an error, update connection status
                connectionIndicator.className = 'connection-indicator disconnected';
                connectionText.textContent = 'Disconnected';
            }
        }
        
        // Function to fetch and update attendance data
        async function fetchAttendanceData() {
            try {
                const dateInput = document.getElementById('attendance-date');
                const date = dateInput.value || new Date().toISOString().split('T')[0];
                
                const response = await fetch(`/api/attendance?date=${date}`);
                const attendanceData = await response.json();
                
                const attendanceList = document.getElementById('attendance-list');
                
                // Clear existing content
                attendanceList.innerHTML = '';
                
                if (attendanceData.length === 0) {
                    attendanceList.innerHTML = '<div class="recognition-entry">No attendance records for the selected date.</div>';
                    return;
                }
                
                // Add attendance entries
                attendanceData.forEach(student => {
                    const item = document.createElement('div');
                    item.className = 'attendance-item';
                    
                    const name = student.name.split('-')[0]; // Strip ID if included in name
                    const initials = name.split(' ').map(n => n[0]).join('');
                    
                    item.innerHTML = `
                        <div class="attendance-avatar">
                            ${initials}
                        </div>
                        <div class="attendance-details">
                            <div class="attendance-name">${name}</div>
                            <div class="attendance-time">${student.time}</div>
                        </div>
                        <div class="attendance-status">${student.status}</div>
                    `;
                    
                    attendanceList.appendChild(item);
                });
            } catch (error) {
                console.error('Error fetching attendance data:', error);
            }
        }
        
        // Setup attendance date picker
        const attendanceDatePicker = document.getElementById('attendance-date');
        attendanceDatePicker.valueAsDate = new Date();
        attendanceDatePicker.addEventListener('change', fetchAttendanceData);
        
        // Reports functionality
        const startDateInput = document.getElementById('start-date');
        const endDateInput = document.getElementById('end-date');
        const generateReportBtn = document.getElementById('generate-report');
        const reportContent = document.getElementById('report-content');
        
        // Set default date range (last 7 days)
        const today = new Date();
        const oneWeekAgo = new Date();
        oneWeekAgo.setDate(today.getDate() - 7);
        
        startDateInput.valueAsDate = oneWeekAgo;
        endDateInput.valueAsDate = today;
        
        generateReportBtn.addEventListener('click', function() {
            const startDate = startDateInput.value;
            const endDate = endDateInput.value;
            
            if (!startDate || !endDate) {
                alert('Please select start and end dates');
                return;
            }
            
            if (new Date(startDate) > new Date(endDate)) {
                alert('Start date must be before end date');
                return;
            }
            
            fetchAttendanceReport(startDate, endDate);
        });
        
        async function fetchAttendanceReport(startDate, endDate) {
            try {
                const response = await fetch(`/api/report?start_date=${startDate}&end_date=${endDate}`);
                const reportData = await response.json();
                
                if (reportData.length === 0) {
                    reportContent.innerHTML = '<p>No attendance data found for the selected date range.</p>';
                    return;
                }
                
                // Create a date range array for the table headers
                const dates = [];
                const start = new Date(startDate);
                const end = new Date(endDate);
                const dateOptions = { month: 'short', day: 'numeric' };
                
                for (let d = new Date(start); d <= end; d.setDate(d.getDate() + 1)) {
                    dates.push(d.toISOString().split('T')[0]);
                }
                
                // Generate the report table
                let tableHTML = `
                    <table class="report-table">
                        <thead>
                            <tr>
                                <th>Student</th>
                `;
                
                // Add date columns
                dates.forEach(date => {
                    const displayDate = new Date(date).toLocaleDateString('en-US', dateOptions);
                    tableHTML += `<th>${displayDate}</th>`;
                });
                
                tableHTML += `
                                <th>Attendance %</th>
                            </tr>
                        </thead>
                        <tbody>
                `;
                
                // Add student rows
                reportData.forEach(student => {
                    const studentName = student.name.split('-')[0]; // Strip ID if included
                    
                    // Determine percentage class
                    let percentageClass = '';
                    if (student.percentage >= 80) {
                        percentageClass = 'high';
                    } else if (student.percentage >= 60) {
                        percentageClass = 'medium';
                    } else {
                        percentageClass = 'low';
                    }
                    
                    tableHTML += `
                        <tr>
                            <td>${studentName}</td>
                    `;
                    
                    // Add status for each date
                    dates.forEach(date => {
                        const status = student.dates[date] || 'Absent';
                        const statusIcon = status === 'Present' ? 
                            '<span style="color: #16a34a;">✓</span>' : 
                            '<span style="color: #dc2626;">✗</span>';
                        tableHTML += `<td>${statusIcon}</td>`;
                    });
                    
                    // Add attendance percentage
                    tableHTML += `
                            <td><span class="attendance-percentage ${percentageClass}">${student.percentage}%</span></td>
                        </tr>
                    `;
                });
                
                tableHTML += `
                        </tbody>
                    </table>
                `;
                
                reportContent.innerHTML = tableHTML;
                
            } catch (error) {
                console.error('Error fetching report data:', error);
                reportContent.innerHTML = '<p>Error generating report. Please try again.</p>';
            }
        }
        
        // Function to update stats
        async function updateStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                
                document.getElementById('total-students').textContent = stats.totalStudents;
                document.getElementById('present-today').textContent = stats.presentToday;
                document.getElementById('current-date').textContent = new Date(stats.date).toLocaleDateString('en-US', { 
                    year: 'numeric', month: 'short', day: 'numeric' 
                });
                document.getElementById('current-time').textContent = stats.time;
                
                // Update connection status
                connectionIndicator.className = 'connection-indicator connected';
                connectionText.textContent = 'Connected';
            } catch (error) {
                console.error('Error updating stats:', error);
                connectionIndicator.className = 'connection-indicator disconnected';
                connectionText.textContent = 'Disconnected';
            }
        }
        
        // Check connection and update data periodically
        function checkConnection() {
            // Set initial state
            connectionIndicator.className = 'connection-indicator connecting';
            connectionText.textContent = 'Connecting...';
            
            // Update stats, events, and attendance data
            updateStats();
            fetchEvents();
            fetchAttendanceData();
        }
        
        // Initialize everything when page loads
        document.addEventListener('DOMContentLoaded', () => {
            checkConnection();
            
            // Check camera state
            toggleMainCamera(false);
            cameraToggle.checked = false;
            cameraFeed.style.display = 'none';
            cameraOffMessage.style.display = 'flex';
            
            // Set up periodic updates
            setInterval(updateStats, 10000); // Every 10 seconds
            setInterval(fetchEvents, 5000); // Every 5 seconds
            setInterval(fetchAttendanceData, 15000); // Every 15 seconds
        });
    </script>
</body>
</html>
        """)

if __name__ == '__main__':
    # Setup templates directory and save HTML
    setup_templates()
    
    # Start face recognition in a background thread
    thread = threading.Thread(target=face_recognition_thread)
    thread.daemon = True
    thread.start()
    
    print("Starting Flask server...")
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)