# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 18:17:04 2020

@author: PJ
"""

#https://www.geeksforgeeks.org/sudoku-backtracking-7/

import numpy as np

def find_empty_location(matrix, l): 
    for row in range(9): 
        for col in range(9): 
            if(matrix[row][col]== 0): 
                l[0]= row 
                l[1]= col 
                return True
    return False

def check_row(matrix, row, num):
    for i in range(9):
        if matrix[row][i] == num:
            return True
    return False

def check_col(matrix, col, num):
    for i in range(9):
        if matrix[i][col] == num:
            return True
    return False

def check_box(matrix, row, col, num):
    for i in range(3):
        for j in range(3):
            if matrix[i + row][j + col] == num:
                return True
    return False

def is_valid(matrix, col, row, num):
    return not check_row(matrix,row,num) and not check_col(matrix,col,num) and not check_box(matrix, row - row % 3, col - col % 3, num) 

def solve_sudoku(matrix):
    l = [0, 0]
    if not find_empty_location(matrix,l):
        return True
    else:
        row = l[0]
        col = l[1]

        for num in range(1,10):
            if is_valid(matrix, col, row, num):
                matrix[row][col] = num
                if(solve_sudoku(matrix)):
                    return True
                else:
                    matrix[row][col] = 0

        return False

if __name__ == "__main__":
    sudoku1 =[[0 for x in range(9)]for y in range(9)] 
    sudoku1 = [[0,0,0,2,6,0,7,0,1],
               [6,8,0,0,7,0,0,9,0],
               [1,9,0,0,0,4,5,0,0],
               [8,2,0,1,0,0,0,4,0],
               [0,0,4,6,0,2,9,0,0],
               [0,5,0,0,0,3,0,2,8],
               [0,0,9,3,0,0,0,7,4],
               [0,4,0,0,5,0,0,3,6],
               [7,0,3,0,1,8,0,0,0]]
    
    sudoku_hard = [[0,2,0,0,0,0,0,0,0],
                   [0,0,0,6,0,0,0,0,3],
                   [0,7,4,0,8,0,0,0,0],
                   [0,0,0,0,0,3,0,0,2],
                   [0,8,0,0,4,0,0,1,0],
                   [6,0,0,5,0,0,0,0,0],
                   [0,0,0,0,1,0,7,8,0],
                   [5,0,0,0,0,9,0,0,0],
                   [0,0,0,0,0,0,0,4,0]]
    
    if solve_sudoku(sudoku_hard):
        print(np.array(sudoku_hard))
    else:
        print("No Solution")