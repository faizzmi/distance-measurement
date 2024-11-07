import csv
import math
import os

import cv2
import cvzone
import numpy as np
from ultralytics import YOLO

def prepare_directories(dir_name):
    root_dir = "output_result"
    video_dir = root_dir+'/videos/'
    frame_dir = root_dir+'/frames/'+dir_name
    car_images_dir = root_dir+'/ori_image/'+dir_name
    excel_file_path = f'{root_dir}/distance_data'

    directories = [root_dir, video_dir, frame_dir, car_images_dir, excel_file_path]

    for directory in directories:
        if not os.path.exists(directory):
            print("Creating " + directory + " directory...")
            os.makedirs(directory)

    return video_dir, frame_dir, car_images_dir, excel_file_path

def resive_window(img):
    width, height = 1280, 720

    # Resize the frame while maintaining aspect ratio
    output_width = width
    aspect_ratio = img.shape[1] / img.shape[0]
    output_height = int(output_width / aspect_ratio)

    reimg = (cv2.resize(img, (output_width, output_height)))

    return reimg, width, height

def display_message(img):
    alpha = 0.4
    overlay = img.copy()

    cv2.rectangle(img,pt1=(50,50),pt2=(250,150),color=(0,0,0),thickness=-1 )
    cv2.rectangle(img, pt1=(1030,50),pt2=(1230,150),color=(0,0,0),thickness=-1)
    cv2.putText(img,f"Status: ", (60,70),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),2)      # Status Message
    cv2.putText(img,f"Nearest Car:",(1040, 70),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),2)# Neareast Car Message
    
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def middlepoint(w,h,image):
    # Calculate middle point
    middle_x = int(w / 2)
    middle_y = int(h)

    # Display the middle point on the bounding box
    cv2.circle(image, (middle_x, middle_y), 6, (0, 0, 255), -1)  # Draw a filled circle

    return middle_x, middle_y

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_color = (255,) * channel_count
    
    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def initialize_yolo_model():
    return YOLO("../Yolo-Weights/yolov8l.pt")

def get_class_names():
    return [
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
        "diningtable", "toilet", "qtvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
    ]

def edges_sobel(image):
    # Apply Sobel operators
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    edges = cv2.normalize(edge_magnitude, None, *(0, 5), cv2.NORM_MINMAX, cv2.CV_8U)
    
    edges = cv2.equalizeHist(edges)

    return edges

def find_contours(edge_image, x1, y1):
    # Find contours coordinate
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)# find maximum contour

    x, y, w, h = cv2.boundingRect(max_contour)# Apply bounding box to max contour

    # re-coordinate
    xn1 = x1 + x
    yn1 = y1 + y
    xn2 = xn1+ (w - x)
    yn2 = yn1 + (h - y)

    return xn1, yn1, xn2, yn2

def center_mark(center_x, center_y, color, img):
    # Mark the center point with an "x" shape
    x_size = 2
    cv2.line(img,
            (center_x - x_size, center_y - x_size),
            (center_x + x_size, center_y + x_size),
            color,
            2
            )
            
    cv2.line(img,
            (center_x + x_size, center_y - x_size),
            (center_x - x_size, center_y + x_size),
            color,
            2
            )

def distance_calculation(bbox_width_pixels):
    focal_length = 683.43
    actual_car_width_meters = 1.67

    # Check if bbox_width_pixels is not zero before performing division
    if bbox_width_pixels != 0:
        distance = (actual_car_width_meters * focal_length) / bbox_width_pixels
        return distance
    else:
        return -1  # Return infinity or any suitable value indicating division by zero

def get_thickness_and_color(distance, current_frame):
    if distance < 5:
        thick = 3
        color = (0, 0, 255) if current_frame % 2 == 0 else (0, 128, 255)  # red or orange
    elif distance < 15:
        thick = 1
        color = (0, 128, 255)  # orange
    else:
        thick = 1
        color = (124, 252, 0)  # green
    
    return thick, color

def get_message(nearest_d, currentframe):
    d = str(nearest_d)
    nearest = d + " m"
    if nearest_d == -1:
        message = "No Car"
        nearest = "-"
        color = (255, 255, 255)
    elif nearest_d < 5:
        message = "Caution"
        color = (0, 0, 255) if currentframe % 2 == 0 else (0, 128, 255)  # red or orange
    elif nearest_d < 15:
        message = "Precaution"
        color = (0, 128, 255)  # orange
    else:
        message = "Normal"
        color = (124, 252, 0)  # green

    return message, nearest, color

def display_text(img, message, nearest, color):
    text_message = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_xm = (50 + 250 - text_message[0]) // 2
    text_ym = (50 + 150 + text_message[1]) // 2

    text_distance = cv2.getTextSize(nearest, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_xd = (1030 + 1230 - text_distance[0]) // 2
    text_yd = (50 + 150 + text_distance[1]) // 2

    # Put the text in the middle of the rectangle
    cv2.putText(img, message, (text_xm, text_ym), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(img, nearest, (text_xd, text_yd), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def process_output_frame(output, distance_list_frame, distance_list, detected_cars, temp, currentframe, frame_dir, car_images_dir, out, number_of_frames):
    cv2.imshow("Output", output)# Display output frame
    # Append data to distance lists
    distance_list_frame.append(temp[:])
    distance_list.append(distance_list_frame[:])
    save_detected_car(detected_cars, currentframe, car_images_dir)# Save detected cars data
    cv2.imwrite("./" + frame_dir + "/frame" + str(currentframe) + ".jpg", output)# Save the processed frame as an image
    out.write(output)# Write the frame into the file 'output.avi'
    print("Processed frame " + str(currentframe) + "/" + str(number_of_frames))# Print processing status

def save_detected_car(detected_cars, currentframe, car_images_dir):
    for i, car_img in enumerate(detected_cars):
        car_image_path = f"{car_images_dir}/frame{currentframe}_car{i}.jpg"
        cv2.imwrite(car_image_path, car_img)
        # print(f"Saved: {car_image_path}")

def save_distance_list_to_excel(distance_list, dir_name, excel_file_path):
    csv_file_path = f'{excel_file_path}/{dir_name}.csv'
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Frame', 'Distance'])
        for row in distance_list:
            csv_writer.writerow(row)
    
    print(f'Saved: {csv_file_path}')

def handle_key_events(video_dir, frame_dir):
    key = cv2.waitKey(1)

    if key & 0xFF == ord("q"):  # q for quitting the video
        video_end(frame_dir, video_dir)
        return True
    
    if key == ord("p"):  # p for pausing the video
        print(f"\nVideo Paused")
        cv2.waitKey(-1)  # Wait until any key is pressed

    return False

def video_end(frame_dir, video_dir):
    print("\nVideo End")

    print(f"Saved: {video_dir}.")
    print(f"Saved: {frame_dir}.")

def process_frame(img, currentframe, distance_list, out, frame_dir, car_images_dir, number_of_frames):
    img, width, height = resive_window(img)
    img = display_message(img)
    middle_x, middle_y = middlepoint(width, height, img)

    # Initialize list
    detected_cars = []
    temp = []
    distance_list_frame = [currentframe]

    region_of_interest_vertices = [
        (0, 0),                       #pt1
        (width, 0),                   #pt2
        (int(7 * width / 8), height), #pt3
        (int(width / 8), height),     #pt4
    ]

    cv2.polylines(img, [np.array(region_of_interest_vertices)], isClosed=True, color=(128,128,128), thickness=1)# Draw the outer line of ROI
    cropped_image = region_of_interest(img, np.array([region_of_interest_vertices], np.int32))# Apply the region of interest mask

    # Process the cropped image
    model = initialize_yolo_model()
    results = model(cropped_image, stream=True)

    for  r in results:#  Identify Vehicles
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])# Class Name
            classNames = get_class_names()
            currentClass = classNames[cls]
            conf = math.ceil((box.conf[0]*100))/100# Confidenece

            if (currentClass == "car" and conf > 0.6 ):#  only "car" is detected

                # Extract coordinates of the bounding box of car
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                car_roi = img[y1:y2, x1:x2]

                blur_car = cv2.GaussianBlur(car_roi,(5,5),0)
                gray_car = cv2.cvtColor(blur_car, cv2.COLOR_BGR2GRAY)
                edge = edges_sobel(gray_car)

                x, y, w, h = find_contours(edge, x1, y1)
                box_width = w - x

                d = distance_calculation(box_width)
                distance = round(d,2)

                if distance >= 0:
                    thick, colour = get_thickness_and_color(distance, currentframe)

                    cv2.rectangle(img,
                                (x, y),
                                (w, h),
                                colour,
                                thick
                            )

                    center_x = (x + w) // 2
                    center_y = (y + h) // 2
                    center_mark(center_x, center_y, (255, 0, 0), img)

                    # # Calculate the angle between the middle point and the center mark
                    # angle_rad = math.atan2(center_y - middle_y, center_x - middle_x)
                    # angle_deg = math.degrees(angle_rad)
                    # angle_text = f"{angle_deg:.0f} degrees"
                    # cv2.putText(img, angle_text, (x, y - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

                    cvzone.putTextRect(img,
                        f"d: {distance:.2f} m",
                        (max(0, x), max(35, y)),
                        scale=0.9,
                        colorR=colour,
                        thickness=2,
                        offset=1
                    )

                    # save detected car in array
                    detected_cars.append(blur_car)
                    temp.append(distance)

            nearest_d = -1  # Default value if no cars are detected
            if temp:  # Check if temp list is not empty
                nearest_d = min(temp)

        message, nearest, color = get_message(nearest_d, currentframe)
        display_text(img, message, nearest, color)
    
    # Inside the main processing loop
    process_output_frame(img, distance_list_frame, distance_list, detected_cars, temp, currentframe, frame_dir, car_images_dir, out, number_of_frames)

    return img

if __name__ == "__main__":
    # Make sure that dir_name same number with video file name!
    cap = cv2.VideoCapture('datasets\dataset_3.mp4')
    
    dir_name = "dataset3"
    currentframe = 0
    distance_list = []

    video_dir, frame_dir, car_images_dir, excel_file_path = prepare_directories(dir_name)

    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Number of frames: ", str(number_of_frames))

    # Define the codec and create VideoWriter object. The output is stored in '.avi' file.
    out = cv2.VideoWriter(os.path.join(video_dir, dir_name + '.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,720))
    
    print("Loading Object Detection...")
    
    while True:
        success, img = cap.read()
        
        if success == True:
            img = process_frame(img, currentframe, distance_list, out, frame_dir, car_images_dir, number_of_frames)

            if handle_key_events(video_dir, frame_dir):# Inside the main processing loop
                break
            
            currentframe += 1
        else:
            video_end(frame_dir, video_dir)
            break
        
    save_distance_list_to_excel(distance_list, dir_name, excel_file_path)

    cap.release()
    cv2.destroyAllWindows()
