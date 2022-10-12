# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by James Gao (jamesjg2@illinois.edu) on 9/03/2021
# Inspired by work done by Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP3
"""

from cmath import sqrt
import math
import numpy as np
from alien import Alien
from typing import List, Tuple

def two_point_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def point_segment_distance(point, segment):
    """Compute the distance from the point to the line segment.
    Hint: Lecture note "geometry cheat sheet"

        Args:
            point: A tuple (x, y) of the coordinates of the point.
            segment: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
     # vector AB
    A = segment[0]
    B = segment[1]
    C = point
    AB = [None, None];
    AB[0] = B[0] - A[0];
    AB[1] = B[1] - A[1];
    # vector BP
    BC = [None, None];
    BC[0] = C[0] - B[0];
    BC[1] = C[1] - B[1];
    # vector AP
    AC = [None, None];
    AC[0] = C[0] - A[0];
    AC[1] = C[1] - A[1];
    # Variables to store dot product
    # Calculating the dot product
    AB_BC = AB[0] * BC[0] + AB[1] * BC[1];
    AB_AC = AB[0] * AC[0] + AB[1] * AC[1];
    # Minimum distance from
    # point E to the line segment
    reqAns = 0;
    # Case 1
    if (AB_BC > 0) :
        # Finding the magnitude
        y = C[1] - B[1];
        x = C[0] - B[0];
        reqAns = np.sqrt(x * x + y * y);
    # Case 2
    elif (AB_AC < 0) :
        y = C[1] - A[1];
        x = C[0] - A[0];
        reqAns = np.sqrt(x * x + y * y);
    # Case 3
    else:
        # Finding the perpendicular distance
        x1 = AB[0];
        y1 = AB[1];
        x2 = AC[0];
        y2 = AC[1];
        mod = np.sqrt(x1 * x1 + y1 * y1);
        reqAns = abs(x1 * y2 - y1 * x2) / mod;
    return reqAns; 

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def do_segments_intersect(segment1, segment2):
    """Determine whether segment1 intersects segment2.  
    We recommend implementing the above first, and drawing down and considering some examples.
    Lecture note "geometry cheat sheet" may also be handy.

        Args:
            segment1: A tuple of coordinates indicating the endpoints of segment1.
            segment2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """
    A = segment1[0]
    B = segment1[1]
    C = segment2[0]
    D = segment2[1]
    if point_segment_distance(A, segment2) == 0:
        return True
    if point_segment_distance(B, segment2) == 0:
        return True
    if point_segment_distance(C, segment1) == 0:
        return True
    if point_segment_distance(D, segment1) == 0:
        return True
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def segment_distance(segment1, segment2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.
    Hint: Distance of two line segments is the distance between the closest pair of points on both.

        Args:
            segment1: A tuple of coordinates indicating the endpoints of segment1.
            segment2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    if do_segments_intersect(segment1, segment2):
        return 0
    else:
        p1 = segment1[0]
        p2 = segment1[1]
        p3 = segment2[0]
        p4 = segment2[1]
        distances = []
        distances.append(point_segment_distance(p1, segment2))
        distances.append(point_segment_distance(p2, segment2))
        distances.append(point_segment_distance(p3, segment1))
        distances.append(point_segment_distance(p4, segment1))
        return min(distances)


def check_circle(center, width, walls, buffer):
    for wall in walls:
        p1 = (wall[0], wall[1])
        p2 = (wall[2], wall[3])
        tangent_dist = point_segment_distance(center, (p1,p2))
        if tangent_dist <= width + buffer:
            return True
    return False

def check_horizontal(alien, walls, buffer):
    head_tail = alien.get_head_and_tail()
    if head_tail[0][0] < head_tail[1][0]:
        head = head_tail[0]
        tail = head_tail[1]
    else:
        head = head_tail[1]
        tail = head_tail[0]
    d = alien.get_width()
    head1 = (head[0], head[1] - d - buffer)
    tail1 = (tail[0], tail[1] - d - buffer)
    head2 = (head[0], head[1] + d + buffer)
    tail2 = (tail[0], tail[1] + d + buffer)
    for wall in walls:
        w1 = (wall[0], wall[1])
        w2 = (wall[2], wall[3])
        if do_segments_intersect((head1, tail1), (w1, w2)) or do_segments_intersect((head2, tail2), (w1, w2)) :
            return True
    if check_circle((head[0], head[1]), d, walls, buffer) or check_circle((tail[0], tail[1]), d, walls, buffer):
        return True
    return False

def check_vertical(alien, walls, buffer):
    head_tail = alien.get_head_and_tail()
    if head_tail[0][1] < head_tail[1][1]:
        head = head_tail[0]
        tail = head_tail[1]
    else:
        head = head_tail[1]
        tail = head_tail[0]
    d = alien.get_width()
    head1 = (head[0] - d - buffer, head[1])
    tail1 = (tail[0] - d - buffer, tail[1])
    head2 = (head[0] + d + buffer, head[1])
    tail2 = (tail[0] + d + buffer, tail[1])
    for wall in walls:
        w1 = (wall[0], wall[1])
        w2 = (wall[2], wall[3])
        if do_segments_intersect((head1, tail1), (w1, w2)) or do_segments_intersect((head2, tail2), (w1, w2)):
            return True
    if check_circle((head[0], head[1]), d, walls, buffer) or check_circle((tail[0], tail[1]), d, walls, buffer):
        return True
    return False


def does_alien_touch_wall(alien, walls, granularity):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format [(startx, starty, endx, endx), ...]
            granularity (int): The granularity of the map

        Return:
            True if touched, False if not
    """
    buffer = granularity / np.sqrt(2)
    if alien.is_circle():
        return check_circle(alien.get_centroid(), alien.get_width(), walls, buffer)
    elif alien.get_shape() == 'Horizontal':
        return check_horizontal(alien, walls, buffer)
    elif alien.get_shape() == 'Vertical':
        return check_vertical(alien, walls, buffer)


def does_alien_touch_goal(alien, goals):
    """Determine whether the alien touches a goal
        
        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            goals (list): x, y coordinate and radius of goals in the format [(x, y, r), ...]. There can be multiple goals
        
        Return:
            True if a goal is touched, False if not.
    """
    if alien.is_circle():
        for goal in goals:
            if two_point_distance(alien.get_centroid(), (goal[0], goal[1])) <= alien.get_width() + goal[2]:
                return True
        return False
    else:
        head_tail = alien.get_head_and_tail()
        head = head_tail[0]
        tail = head_tail[1]
        d = alien.get_width()
        for goal in goals:
            goal_center = (goal[0], goal[1])
            radius = goal[2]
            if two_point_distance(goal_center, head) <= radius + d:
                return True
            if two_point_distance(goal_center, tail) <= radius + d:
                return True
            walls = []
            if alien.get_shape() == 'Horizontal':
                walls = [(head[0], head[1] - d, tail[0], tail[1] - d), (head[0], head[1] + d, tail[0], tail[1] + d)]
            if alien.get_shape() == 'Vertical':
                walls = [(head[0] - d, head[1], tail[0] - d, tail[1]), (head[0] + d, head[1], tail[0] + d, tail[1])]
            if check_circle(goal_center, radius, walls, 0):
                return True
        return False

def is_alien_within_window(alien, window,granularity):
    """Determine whether the alien stays within the window
        
        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
            granularity (int): The granularity of the map
    """
    w = window[0]
    h = window[1]
    walls = [(0,0,w,0), (0,0,0,h), (0,h,w,h), (w,0,w,h)]
    buffer = granularity / np.sqrt(2)
    if alien.is_circle():
        return not check_circle(alien.get_centroid(), alien.get_width(), walls, buffer)
    elif alien.get_shape() == 'Horizontal':
        return not check_horizontal(alien, walls, buffer)
    elif alien.get_shape() == 'Vertical':
        return not check_vertical(alien, walls, buffer)


if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result

    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                  f'{b} is expected to be {result[i]}, but your' \
                                                                  f'result is {distance}'

    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls, 0)
        touch_goal_result = does_alien_touch_goal(alien, goals)
        in_window_result = is_alien_within_window(alien, window, 0)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert touch_goal_result == truths[
            1], f'does_alien_touch_goal(alien, goals) with alien config {config} returns {touch_goal_result}, ' \
                f'expected: {truths[1]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    # Initialize Aliens and perform simple sanity check.
    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")