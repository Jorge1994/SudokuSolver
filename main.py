# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 17:49:09 2020

@author: PJ
"""
import cv2
import SudokuExtractor
import keras

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720) 
    model = keras.models.load_model('digit_model.h5')
    old_sudoku = None
    while True:
        ret, frame = cap.read()
        if ret == True:
            new_frame, old_sudoku = SudokuExtractor.extract_and_solve_sudoku(frame, model, old_sudoku)
            cv2.imshow("Solved", new_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()