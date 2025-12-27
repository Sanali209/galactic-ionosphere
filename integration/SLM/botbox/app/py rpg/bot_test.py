import time
import unittest
import random
import cv2
import numpy as np
import pygame
from PIL import Image

from SLM.botbox.Environment import Vector2d, botTtrackEntity, Path
from SLM.botbox.behTree import ActionNode, Sequence, Blackboard, Selector, Inverter
from behavior_funct import is_enemy_near, evade, move_by_patrol_points, find_collectable_resources, \
    get_next_patrol_point, move_to_target, check_pos_reached, select_new_position


class DebugEnv:
    def __init__(self):
        pygame.init()
        self.game_screen_size = 800, 600
        self.board_screen = pygame.display.set_mode(self.game_screen_size)

        self.font_size = 16
        self.font = pygame.font.Font(None, self.font_size)
        self.stop = False

        self.blackboard = Blackboard()
        self.blackboard.set_value("current_position", Vector2d(100, 100))
        self.blackboard.set_value("target_position", Vector2d(200, 200))
        patrol_path = Path()
        patrol_path.path.extend([Vector2d(200, 200), Vector2d(300, 300)])
        self.blackboard.set_value("patrol_path", Path)
        ent = []
        for i in range(10):
            ent.append(Colectable_weed())
        self.blackboard.set_value("ent_track",ent)




        is_enemy_near_node = ActionNode(is_enemy_near)
        evade_node = ActionNode(evade)
        move_by_patrol_points_node = ActionNode(move_by_patrol_points)
        find_collectable_resources_node = ActionNode(find_collectable_resources)
        collect_resources_node = ActionNode(find_collectable_resources)
        is_backpack_full_node = ActionNode(find_collectable_resources)
        move_to_base_node = ActionNode(find_collectable_resources)
        point_reached_node = ActionNode(check_pos_reached)

        self.bh_tree = Sequence()
        selector = Selector()
        selector.add_child(Inverter(ActionNode(check_pos_reached)))
        selector.add_child(ActionNode(select_new_position))
        self.bh_tree.add_child(selector)
        self.bh_tree.add_child(ActionNode(move_to_target))

    def run(self):
        while not self.stop:
            self.bh_tree.run(self.blackboard)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.stop = True

            self.board_screen.fill((0, 0, 0))
            #draw player position
            p_pos:Vector2d = self.blackboard.get_value("current_position")
            pygame.draw.circle(self.board_screen,(0,255,0),p_pos.to_tuple(),
                               3.0)
            target_pos:Vector2d = self.blackboard.get_value("target_position")
            pygame.draw.circle(self.board_screen,(255,0,0),target_pos.to_tuple(),
                               3.0)
            for ent_i in self.blackboard.get_value('ent_track'):
                pygame.draw.circle(self.board_screen, (0, 0, 255), ent_i.position.to_tuple(),
                                   3.0)

            pygame.display.flip()
            time.sleep(0.1)

    def behaviorTree(self):
        # develop bot behavior
        # all time look if enemy is near evade
        # 1. move by patrol points
        # 2. point rich find collectable resources
        # 3.collect resources and look for backpack is full if full go to base
        is_enemy_near_node = ActionNode(is_enemy_near)
        evade_node = ActionNode(evade)
        move_by_patrol_points_node = ActionNode(move_by_patrol_points)
        find_collectable_resources_node = ActionNode(find_collectable_resources)
        collect_resources_node = ActionNode(find_collectable_resources)
        is_backpack_full_node = ActionNode(find_collectable_resources)
        move_to_base_node = ActionNode(find_collectable_resources)
        point_reached_node = ActionNode(point_reached)

        # move by patrol points sequence
        move_by_patrol_points_sequence = Sequence()
        move_by_patrol_points_sequence.add_child(point_reached_node)
        move_by_patrol_points_sequence.add_child(ActionNode(get_next_patrol_point))
        move_by_patrol_points_sequence.add_child(ActionNode(move_to_target))

        # create a sequence for normal behavior
        normal_behavior = Sequence()
        normal_behavior.add_child(move_by_patrol_points_sequence)
        normal_behavior.add_child(find_collectable_resources_node)
        normal_behavior.add_child(collect_resources_node)
        # create a sequence for evading enemy
        evade_sequence = Sequence()
        evade_sequence.add_child(is_enemy_near_node)
        evade_sequence.add_child(evade_node)

        # create a sequence for moving to base
        move_to_base_sequence = Sequence()
        move_to_base_sequence.add_child(is_backpack_full_node)
        move_to_base_sequence.add_child(move_to_base_node)

        # create a sequence for main behavior
        main_sequence = Sequence()
        main_sequence.add_child(evade_sequence)
        main_sequence.add_child(normal_behavior)
        main_sequence.add_child(move_to_base_sequence)

        return main_sequence


class Colectable_weed(botTtrackEntity):
    def __init__(self):
        super().__init__()
        self.class_name = "Colectable_weed"
        self.position = Vector2d(random.randint(10, 800), random.randint(10, 600))
        self.collect_time = 10


class Enemy(botTtrackEntity):
    def __init__(self):
        super().__init__()
        self.class_name = "Enemy"
        self.position = Vector2d(random.randint(10, 800), random.randint(10, 600))


def draw_point(cv_image, pos, color):
    cv2.circle(cv_image, (int(pos.x), int(pos.y)), 5, color, -1)


def draw_line(cv_image, start, end, color):
    cv2.line(cv_image, (int(start.x), int(start.y)), (int(end.x), int(end.y)), color, 2)


def draw_wire_circle(cv_image, pos, radius, color):
    cv2.circle(cv_image, (int(pos.x), int(pos.y)), radius, color, 1)


class TestBot(unittest.TestCase):

    def test_vector(self):
        canvas_size = Vector2d(640, 480)
        player_pos = Vector2d(canvas_size.x / 2, canvas_size.y / 2)
        target_pos = Vector2d(player_pos.x - 100, player_pos.y - 100)

        move_dir = target_pos - player_pos
        move_dir = move_dir.normalize()

        print(move_dir)
        player_steer = move_dir * 100
        player_steer = player_steer + player_pos

        cv_image = np.zeros((480, 640, 3), np.uint8)
        draw_point(cv_image, player_pos, (0, 255, 0))
        draw_point(cv_image, target_pos, (0, 0, 255))

        draw_line(cv_image, player_pos, player_steer, (255, 255, 255))

        pil_image = Image.fromarray(cv_image)

        # save image
        pil_image.save("test_vector.png")

    def test_minimap_detect(self):
        cv_image = cv2.imread("screenshots/1723945738.74109.png")
        mini_map_rect = [1620, 810, 1900, 1040]
        mini_map = cv_image[mini_map_rect[1]:mini_map_rect[3], mini_map_rect[0]:mini_map_rect[2]]
        mini_map = cv2.cvtColor(mini_map, cv2.COLOR_BGR2RGB)
        target_rgb = [97, 179, 255]
        matches = np.where(np.all(mini_map == target_rgb, axis=-1))
        if len(matches[0]) > 0:
            print("Found target color")
        mean_x = int(np.mean(matches[0]))
        mean_y = int(np.mean(matches[1]))
        x_pos = matches[0][0]
        y_pos = matches[1][0]
        print(x_pos, y_pos)
        print(mean_x, mean_y)

    def test_render(self):
        de = DebugEnv()
        de.run()
