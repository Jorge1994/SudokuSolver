"""
Sudoku Solver using Backtracking Algorithm.

This module implements a recursive backtracking algorithm to solve 9x9 Sudoku puzzles.
The algorithm tries numbers 1-9 in empty cells, checking Sudoku constraints (row, column, 
and 3x3 box uniqueness) at each step. If a conflict is detected, it backtracks and tries 
a different number.

Reference: Backtracking algorithm for Sudoku solving
"""

import numpy as np

# Sudoku grid constants
GRID_SIZE = 9
BOX_SIZE = 3
EMPTY_CELL = 0
MIN_DIGIT = 1
MAX_DIGIT = 9


def find_empty_cell(grid):
    """
    Find the first empty cell in the Sudoku grid.

    Args:
        grid (list): 9x9 2D list representing the Sudoku grid, where 0 indicates an empty cell.

    Returns:
        tuple or None: (row, col) tuple of the first empty cell, or None if grid is complete.
    """
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if grid[row][col] == EMPTY_CELL:
                return (row, col)
    return None


def is_solved(grid):
    """
    Check if the Sudoku puzzle is completely solved.

    Args:
        grid (list): 9x9 2D list representing the Sudoku grid.

    Returns:
        bool: True if all cells are filled (no zeros), False otherwise.
    """
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if grid[row][col] == EMPTY_CELL:
                return False
    return True


def is_number_in_row(grid, row, number):
    """
    Check if a number already exists in the specified row.

    Args:
        grid (list): 9x9 2D list representing the Sudoku grid.
        row (int): Row index to check (0-8).
        number (int): Number to search for (1-9).

    Returns:
        bool: True if number exists in the row, False otherwise.
    """
    for col in range(GRID_SIZE):
        if grid[row][col] == number:
            return True
    return False


def is_number_in_column(grid, column, number):
    """
    Check if a number already exists in the specified column.

    Args:
        grid (list): 9x9 2D list representing the Sudoku grid.
        column (int): Column index to check (0-8).
        number (int): Number to search for (1-9).

    Returns:
        bool: True if number exists in the column, False otherwise.
    """
    for row in range(GRID_SIZE):
        if grid[row][column] == number:
            return True
    return False


def is_number_in_box(grid, box_start_row, box_start_col, number):
    """
    Check if a number already exists in the specified 3x3 box.

    Args:
        grid (list): 9x9 2D list representing the Sudoku grid.
        box_start_row (int): Starting row index of the 3x3 box (0, 3, or 6).
        box_start_col (int): Starting column index of the 3x3 box (0, 3, or 6).
        number (int): Number to search for (1-9).

    Returns:
        bool: True if number exists in the 3x3 box, False otherwise.
    """
    for row_offset in range(BOX_SIZE):
        for col_offset in range(BOX_SIZE):
            if grid[box_start_row + row_offset][box_start_col + col_offset] == number:
                return True
    return False


def is_valid_placement(grid, row, col, number):
    """
    Check if placing a number at the specified position is valid according to Sudoku rules.

    A placement is valid if the number doesn't already exist in:
    - The same row
    - The same column
    - The same 3x3 box

    Args:
        grid (list): 9x9 2D list representing the Sudoku grid.
        row (int): Row index for placement (0-8).
        col (int): Column index for placement (0-8).
        number (int): Number to place (1-9).

    Returns:
        bool: True if placement is valid, False otherwise.
    """
    # Calculate the top-left corner of the 3x3 box containing this cell
    box_start_row = row - (row % BOX_SIZE)
    box_start_col = col - (col % BOX_SIZE)
    
    return (not is_number_in_row(grid, row, number) and 
            not is_number_in_column(grid, col, number) and 
            not is_number_in_box(grid, box_start_row, box_start_col, number))


def solve_sudoku(grid):
    """
    Solve a Sudoku puzzle using recursive backtracking algorithm.

    This function modifies the input grid in place. It recursively attempts to fill
    empty cells with valid numbers (1-9). If a dead-end is reached, it backtracks
    by resetting the cell to 0 and trying a different number.

    Args:
        grid (list): 9x9 2D list representing the Sudoku grid, where 0 indicates empty cells.

    Returns:
        tuple: (bool, list) - (True, solved_grid) if solution found, 
                              (False, grid) if no solution exists.
    """
    # Find the next empty cell
    empty_cell = find_empty_cell(grid)
    
    # If no empty cells remain, puzzle is solved
    if empty_cell is None:
        return True, grid
    
    row, col = empty_cell
    
    # Try numbers 1 through 9
    for number in range(MIN_DIGIT, MAX_DIGIT + 1):
        if is_valid_placement(grid, row, col, number):
            # Place the number
            grid[row][col] = number
            
            # Recursively attempt to solve the rest of the puzzle
            is_solution_found, _ = solve_sudoku(grid)
            
            if is_solution_found:
                return True, grid
            
            # Backtrack: reset cell and try next number
            grid[row][col] = EMPTY_CELL
    
    # No valid number found for this cell
    return False, grid