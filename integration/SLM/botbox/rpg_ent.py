import pygame
from pygame.sprite import Sprite

from SLM.botbox.Environment import Pawn, EntityController


class Actor(Pawn):
    def __init__(self):
        super().__init__()
        self.actor_size = (64, 64)
        self.name = 'actor'
        self.actor_sprite = Sprite()
        self.stats = Stats()
        self.key_color = (0, 0, 0)
        self.forward = [0, 0]

    def update(self):
        transform = self.transform
        camera = self.env.renderer.camera
        self.actor_sprite.rect.x = transform.position[0] - camera.position[0]
        self.actor_sprite.rect.y = transform.position[1] - camera.position[1]
        super().update()

    def load_sprite(self, sprite_path):
        self.actor_sprite.image = pygame.image.load(sprite_path)
        self.actor_sprite.image = pygame.transform.scale(self.actor_sprite.image, self.actor_size)
        self.actor_sprite.image.convert()
        self.actor_sprite.image.set_colorkey(self.key_color)
        self.actor_sprite.rect = self.actor_sprite.image.get_rect()

    def loadSpriteFromSheet(self, sheet_path, position, size):
        sheet = pygame.image.load(sheet_path)
        self.actor_sprite.image = get_tile(sheet, position[0], position[1], size)
        self.actor_sprite.image = pygame.transform.scale(self.actor_sprite.image, self.actor_size)
        self.actor_sprite.image.convert()
        self.actor_sprite.image.set_colorkey(self.key_color)
        self.actor_sprite.rect = self.actor_sprite.image.get_rect()

    def on_add_parent_change(self, old_parent, new_parent):
        if new_parent:
            new_parent.sprite_groups['npc'].add(self.actor_sprite)
        if old_parent:
            old_parent.sprite_groups['npc'].remove(self.actor_sprite)

class Stats:
    def __init__(self):
        self.hp = 100
        self.mp = 100
        self.stamina = 100
        self.strength = 10
        self.defense = 10
        self.magic = 10
        self.speed = 10
        self.luck = 10
        self.speed = 50
        self.attack_range = 3


class Player(Actor):
    def __init__(self):
        super().__init__()
        self.name = 'player'
        self.loadSpriteFromSheet('res/Aesthetic pack base set (Free)/Player Sample.png', (0, 0), (24, 36))
        controller = PlayerController()
        self.AddController(controller)

    def on_add_parent_change(self, old_parent, new_parent):

        if new_parent:
            new_parent.sprite_groups['player'].add(self.actor_sprite)
        if old_parent:
            old_parent.sprite_groups['player'].remove(self.actor_sprite)


class Obstacle(Actor):
    def __init__(self):
        super().__init__()
        self.name = 'obstacle'
        self.load_sprite('res/rock_obstacle.png')

    def on_add_parent_change(self, old_parent, new_parent):
        if new_parent:
            new_parent.sprite_groups['obstacle'].add(self.actor_sprite)
        if old_parent:
            old_parent.sprite_groups['obstacle'].remove(self.actor_sprite)



class PlayerController(EntityController):
    def __init__(self):
        super().__init__()

    def update(self):
        camera = self.env.renderer.camera
        camera.position = [self.parentEntity.transform.position[0] - 320, self.parentEntity.transform.position[1] - 240]
        need_update, last_time = super().update()
        keys = pygame.key.get_pressed()
        transform = self.parentEntity.transform
        speed = self.parentEntity.stats.speed
        key_pressed = False
        if keys[pygame.K_LEFT]:
            self.parentEntity.forward[0] = -1
            key_pressed = True

        if keys[pygame.K_RIGHT]:
            self.parentEntity.forward[0] = 1
            key_pressed = True
        if not keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
            self.parentEntity.forward[0] = 0
        if keys[pygame.K_UP]:
            self.parentEntity.forward[1] = -1
            key_pressed = True
        if keys[pygame.K_DOWN]:
            self.parentEntity.forward[1] = 1
            key_pressed = True
        if not keys[pygame.K_UP] and not keys[pygame.K_DOWN]:
            self.parentEntity.forward[1] = 0
        if key_pressed:
            transform.position[0] += self.parentEntity.forward[0] * (speed * last_time)
            transform.position[1] += self.parentEntity.forward[1] * (speed * last_time)
        separate_vector = self.check_obstacle_collision()
        if separate_vector:
            transform.position[0] += separate_vector[0] * (speed * last_time)
            transform.position[1] += separate_vector[1] * (speed * last_time)

    def check_obstacle_collision(self):
        obstacles = self.parentEntity.parentEntity.sprite_groups['obstacle']
        player_sprite = self.parentEntity.actor_sprite
        collide = pygame.sprite.spritecollideany(player_sprite, obstacles)
        if collide:
            return self.calc_separate_vector(collide)
        return None

    def calc_separate_vector(self, obstacle):

        player_sprite = self.parentEntity.actor_sprite
        player_rect = player_sprite.rect
        obstacle_rect = obstacle.rect
        player_center = player_rect.center
        obstacle_center = obstacle_rect.center
        separate_vector = [0, 0]
        if player_center[0] < obstacle_center[0]:
            separate_vector[0] = -1
        else:
            separate_vector[0] = 1
        if player_center[1] < obstacle_center[1]:
            separate_vector[1] = -1
        else:
            separate_vector[1] = 1
        return separate_vector

