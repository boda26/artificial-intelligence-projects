# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
import heapq
from os import curdir

# Search should return the path and the number of states explored.
# The path should be a list of MazeState objects that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (astar)
# You may need to slight change your previous search functions in MP2 since this is 3-d maze
def search(maze, searchMethod):
    return {
        "astar": astar,
    }.get(searchMethod, [])(maze)

def astar(maze, ispart1=False):
    """
    This function returns an optimal path in a list, which contains the start and objective.

    @param maze: Maze instance from maze.py
    @param ispart1:pass this variable when you use functions such as getNeighbors and isObjective. DO NOT MODIFY THIS
    @return: a path in the form of a list of MazeState objects
    """
    # Your code here
    starting_state = maze.getStart()
    visited_states = {hash(starting_state): (None, 0)}
    frontier = []
    heapq.heappush(frontier, starting_state)
    while len(frontier) != 0:
        top = heapq.heappop(frontier)
        if top.is_goal():
            path = backtrack(visited_states, top)
            return path
        for state in top.get_neighbors(ispart1=True):
            if hash(state) not in visited_states or state.dist_from_start + state.h < visited_states[hash(state)][1]:
                visited_states[hash(state)] = (top, state.dist_from_start + state.h)
                heapq.heappush(frontier, state)
    return None

# This is the same as backtrack from MP2
def backtrack(visited_states, current_state):
    path = []
    # Your code here ---------------
    path.append(current_state)
    cur = current_state
    while cur.dist_from_start != 0:
        for key, value in visited_states.items():
            if key == hash(cur):
                path.append(value[0])
                cur = value[0]
    path.reverse()
    # ------------------------------
    return path
        