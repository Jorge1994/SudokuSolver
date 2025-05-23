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

def is_solved(matrix):
    for i in range(9):
        for j in range(9):
            if(matrix[i][j] == 0):
                return False
    return True

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
        return True, matrix
    else:
        row = l[0]
        col = l[1]

        for num in range(1,10):
            if is_valid(matrix, col, row, num):
                matrix[row][col] = num
                solved, _ = solve_sudoku(matrix)
                if(solved):
                    return True, matrix
                else:
                    matrix[row][col] = 0

        return False, matrix