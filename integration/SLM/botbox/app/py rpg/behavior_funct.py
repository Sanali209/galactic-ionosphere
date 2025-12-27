import math
import random

from SLM.botbox.Environment import Vector2d
from bot_actions import RPGMouseSteering

def check_pos_reached(blackboard):
    tolerasnce = 1
    current_position = blackboard.get_value('current_position')
    target_position = blackboard.get_value('target_position')

    # Check if the target has been reached
    if abs(current_position.x - target_position.x) < tolerasnce and abs(current_position.y - target_position.y) < tolerasnce:
        return "success"
    return "failure"

def select_new_position(blackboard):
    new_position = Vector2d(random.randint(10, 200), random.randint(10, 200))
    blackboard.set_value('target_position', new_position)
    return "success"


def get_next_patrol_point(blackboard):
    return "success"

def move_to_target(blackboard):
    current_position = blackboard.get_value('current_position')
    target_position = blackboard.get_value('target_position')

    dir:Vector2d = target_position-current_position
    dir = dir.normalize()
    new_pos = current_position+dir
    blackboard.set_value('current_position',new_pos)

def move_to_target_samp(blackboard):
    target = blackboard.get_value("target_position")
    if target:
        print(f"Moving to {target}")
        return "success"
    return "failure"

def move_to_target_m(blackboard):
    dist_tolerance = 0.5
    # Get the current position and target position from the blackboard
    current_position = blackboard.get_value('current_position')
    target_position = blackboard.get_value('target_position')
    if current_position is None or target_position is None:
        return "failure"
    # Calculate the distance to the target
    distance_x = target_position[0] - current_position[0]
    distance_y = target_position[1] - current_position[1]

    # If the distance is small enough, we consider the movement complete
    if abs(distance_x) < dist_tolerance and abs(distance_y) < dist_tolerance:
        return "success"

    # find move direction vector and normalize it
    # Calculate the movement direction vector
    move_dir = [distance_x, distance_y]

    # Normalize the movement vector
    vector_length = math.sqrt(move_dir[0] ** 2 + move_dir[1] ** 2)
    if vector_length != 0:
        move_dir[0] /= vector_length
        move_dir[1] /= vector_length

    action = RPGMouseSteering('move')
    action.movement_vector = move_dir

    # Update the blackboard with the normalized move direction
    blackboard.set_value('action', action)

def is_enemy_near(blackboard):
    #action for checking if enemy is near
    min_enemy_distance = 100
    enemy_list = blackboard.get("enemy_list")
    pass

def evade(blackboard):
    #action for evading enemy
    pass


def move_by_patrol_points( blackboard):
    # Implement logic to move by patrol points
    pass


def find_collectable_resources( blackboard):
    # Implement logic to find collectable resources
    pass

def is_backpack_full( blackboard):
    # Implement logic to check if backpack is full
    pass

def collect_resources( blackboard):
    # Implement logic to collect resources and check if backpack is full
    pass

def move_to_base( blackboard):
    # Implement logic to move back to base if backpack is full
    pass

def check_health(blackboard):
    health = blackboard.get_value("health")
    return "success" if health > 50 else "failure"



def attack(blackboard):
    print("Attacking the enemy!")
    return "success"