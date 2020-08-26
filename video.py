#!/usr/bin/env python3 
import cv2

def video_to_frames( image_path, video_path ):

    videoCap = cv2.VideoCapture( video_path )

    success, image = videoCap.read() 
    count = 1 
    
    while success:
        
        imageName = image_path + 'frame' + count + '.jpg' 
        cv2.imwrite( imageName, image )
        
        success, images = videoCap.read() 

        if count % 1000 or count == 20399: 
            print( "Read new frame: ", count )
        
        count += 1 
        imageName = 0 
