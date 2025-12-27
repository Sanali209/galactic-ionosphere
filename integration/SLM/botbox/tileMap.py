import math
import random

import pygame
from pygame.sprite import Sprite

from SLM.botbox.Environment import EnvironmentEntity, EntityController

class TileMap(EnvironmentEntity):
    def __init__(self):
        super().__init__()
        self.tile_size = (16, 16)
        self.map_size = (640, 480)
        self.tiles = []
        self.tiles_set = []
        grass_tile = Tile()
        grass_tile.load_tile('GRASS+.png', self.tile_size, 0, 0)
        self.addTile(grass_tile)
        grass_tile = Tile()
        grass_tile.load_tile('GRASS+.png', self.tile_size, 1, 0)
        self.addTile(grass_tile)
        grass_tile = Tile()
        grass_tile.load_tile('GRASS+.png', self.tile_size, 2, 0)
        self.addTile(grass_tile)
        self.sprite_groups = {"tiles": pygame.sprite.Group(),
                              "player": pygame.sprite.Group(),
                              "npc": pygame.sprite.Group(),
                              "obstacle": pygame.sprite.Group(),
                              }
        render_entity = PyGameTileMapRenderer()
        self.AddController(render_entity)

    def addTile(self, tile):
        self.tiles_set.append(tile)

    def GenerateTiles(self, width, height):
        self.map_size = (width, height)
        for x in range(0, width):
            self.tiles.append([])
            for y in range(0, height):
                self.tiles[x].append(random.randint(0, len(self.tiles_set) - 1))


class Tile:
    def __init__(self):
        super().__init__()
        self.tile_sprite = Sprite()
        self.obstacle = False

    def load_tile(self, tile_path, size, x, y):
        tile_sheet = pygame.image.load(tile_path)
        self.tile_sprite.image = get_tile(tile_sheet, x, y, size)
        self.tile_sprite.image = pygame.transform.scale(self.tile_sprite.image, size)
        self.tile_sprite.image.convert()
        self.tile_sprite.image.set_colorkey((254, 254, 254))
        self.tile_sprite.rect = self.tile_sprite.image.get_rect()

    def get_copy(self):
        new_tile = Tile()
        new_tile.tile_sprite = self.tile_sprite
        return new_tile

class PyGameTileMapRenderer(EntityController):
    def __init__(self):
        super().__init__()

    def render(self):
        tile_map = self.parentEntity
        camera = self.env.renderer.camera
        self.update_tile_map(tile_map, camera)
        for sprite_group in tile_map.sprite_groups.values():
            self.env.renderer.draw_sprite_group(sprite_group)

    def update_tile_map(self, tile_map, camera):
        tiles_draw_group = tile_map.sprite_groups['tiles']
        tiles_draw_group.empty()
        tile_size = tile_map.tile_size
        tiles = tile_map.tiles
        tiles_set = tile_map.tiles_set
        camera_position = camera.position
        start_x = math.ceil(camera_position[0] // tile_size[0])
        start_y = math.ceil(camera_position[1] // tile_size[1])
        if start_x < 0:
            start_x = 0
        if start_y < 0:
            start_y = 0
        end_x = start_x + (camera_position[0] + camera.view_size[0]) // tile_size[0]
        end_y = start_y + (camera_position[1] + camera.view_size[1]) // tile_size[1]
        end_x = math.ceil(end_x) + 1
        end_y = math.ceil(end_y) + 1

        for x in range(start_x, end_x):
            for y in range(start_y, end_y):
                tile_index = tiles[x][y]
                tile = tiles_set[tile_index]
                tile.tile_sprite.rect.x = x * tile_size[0] - camera_position[0]
                tile.tile_sprite.rect.y = y * tile_size[1] - camera_position[1]
                self.env.renderer.draw_sprite(tile.tile_sprite)
