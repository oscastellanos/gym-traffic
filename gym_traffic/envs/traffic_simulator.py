# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 15:54:28 2018

@author: Osvaldo Castellanos
"""

# !/usr/bin/env python
import os, pygame
import numpy as np
from pygame.locals import *
import sys
import time


sys.path.append('../')

main_dir = os.path.split(os.path.abspath(__file__))[0]

MEAN = 30
STD = 10
CYCLE = 15

# Default Car drives horizontally
class Car:
    def __init__(self, location, height, speed, screen):
        self.stop = True
        self.counter = int(np.random.normal(MEAN, STD))
        self.no_collision = True
        self.speed = speed
        #pygame.sprite.Sprite.__init__(self)
        car = load_image('square.png').convert()
        self.image = pygame.transform.scale(car, (15, 10))
        self.screen = screen
        self.pos = self.image.get_rect().move(location, height)
        self.stationary = False
        self.velocity = 0
        self.waiting = 0
        self.moving = True
        #pygame.Rect.__init__()

    def move(self):
        self.pos = self.pos.move(self.speed, 0)

    def position(self):
        return self.pos

    def idle(self):
        self.stationary = True
        self.waiting += 1
        self.velocity = 0
        self.moving = False

    def resume_driving(self):
        self.stationary = False
        self.waiting = 0
        self.velocity = self.speed
        self.moving = True

    def draw(self):
        self.screen.blit(self.image, self.pos)

    def __str__(self):
        return(str([self.pos.right, self.pos.left, self.pos.top, self.pos.bottom]))

class CarVertical(Car):
    def __init__(self, location, height, speed, screen):
        Car.__init__(self, location, height, speed, screen)
        car = load_image('square-vertical.png').convert()
        self.image = pygame.transform.scale(car, (10, 15))

    def move(self):
        self.pos = self.pos.move(0, self.speed)

class Lane():
    def __init__(self, screen, background):
        self.mean = 30
        self.std = 10
        self.cycle = 15
        self.vehicles_driving = []
        self.red_counter = 0
        self.count = 0
        self.num_of_cars = .95
        self.screen = screen
        self.background = background
        self.collision = False
        self.in_bounds = []
        self.reward_sum = 0
        self.intersection = []
        self.position_matrix = []
        self.velocity_matrix = []
        self.count = 0
    def generateVehicles(self):
        pass
    def draw(self):
        for _ in self.vehicles_driving:
            _.draw()
    def blit_background(self, position1, position2):
        self.screen.blit(self.background, position1, position2)
    def collision(self):
        return self.collision
    # First create the in bounds queue
    def _create_boundary(self, car, inner, outer):
        if (car.pos.left >= outer) & (car.pos.right <= inner):
            if car not in self.in_bounds:
                self.in_bounds.append(car)
                self.count += 1
                #print("This is the car: " + str(self.count))
        else:
            if car in self.in_bounds:
                self.in_bounds.remove(car)
                self.count -= 1
        
        # Next, create the position, velocity matrices
    def _create_matrices(self):
        self.position_matrix = np.zeros((1, 7))
        self.velocity_matrix = np.zeros((1, 7))
        for car in self.in_bounds:
            right = car.pos.right
            if (right > 171) & (right < 188):
                self.position_matrix[0][0] = 1
                self.velocity_binary(car, 0, 0)
                # self.velocity(car, 0, 0)
            elif (right > 187) & (right < 204):
                self.position_matrix[0][1] = 1
                self.velocity_binary(car, 0, 1)
                # self.velocity(car, 0, 1)
            elif (right > 203) & (right < 220):
                self.position_matrix[0][2] = 1
                self.velocity_binary(car, 0, 2)
                # self.velocity(car, 0, 2)
            elif (right > 219) & (right < 236):
                self.position_matrix[0][3] = 1
                self.velocity_binary(car, 0, 3)
                # self.velocity(car, 0, 3)
            elif (right > 235) & (right < 252):
                self.position_matrix[0][4] = 1
                self.velocity_binary(car, 0, 4)
                # self.velocity(car, 0, 4)
            elif (right > 251) & (right < 268):
                self.position_matrix[0][5] = 1
                self.velocity_binary(car, 0, 5)
                # self.velocity(car, 0, 5)
            elif (right > 267) & (right < 285):
                self.position_matrix[0][6] = 1
                self.velocity_binary(car, 0, 6)
                # self.velocity(car, 0, 6)
        #return self.position_matrix, self.velocity_matrix
    def _get_state(self, inner, outer):
        # Create the boundary that we care about to get a position and velocity matrix for
        for o in self.vehicles_driving:
           self._create_boundary(o, inner, outer)

        self._create_matrices()
        return self.position_matrix #self.velocity_matrix
    
    def _get_reward(self):
        for car in self.vehicles_driving:
            self.reward_sum += car.waiting
        return self.reward_sum
    def return_intersection(self):
        return self.intersection
    def reset(self):
        self.vehicles_driving = []
        self.position_matrix = []
        self.velocity_matrix = []
        self.in_bounds = []
        self.reward_sum = 0

    def velocity(self, car, i, j):
        '''
        This velocity method sets the velocity cell to be equal to the current vehicle's velocity.
        :param car:
        :param i:
        :param j:
        :return:
        '''
        self.velocity_matrix[i][j] = car.velocity
    def velocity_binary(self, car, i, j):
        '''
        Velocity binary is the method where the velocity cell is 1 if the car is currently moving, 0 otherwise.
        :param car:
        :param i:
        :param j:
        :return:
        '''
        if car.moving:
            self.velocity_matrix[i][j] = 1
        else:
            self.velocity_matrix[i][j] = 0

class WestLane(Lane):
    def __init__(self, screen, background):
        Lane.__init__(self, screen, background)

    def generateVehicles(self):
        if np.random.rand() > self.num_of_cars:
            occupy = False
            for _ in self.vehicles_driving:
                if _.pos.left >= 607:
                    occupy = True
                    break
            if occupy == False:
                self.vehicles_driving.append(Car(600, 370, speed=-2, screen=self.screen))

    def update(self, light):
        inbound_vehicles_driving = []
        for o in self.vehicles_driving:
            self.screen.blit(self.background, o.pos, o.pos)
            if ((o.pos.right > 355) & (o.pos.right < 358) & ((light == 1) | (light == 2) | (light == 4))):
                o.idle()
            else:
                self.collision = False
                for e in self.vehicles_driving:
                    if (o.pos.left + o.speed < e.pos.right) & (o.pos.left > e.pos.left):
                        self.collision = True
                        o.idle()
                        break
                if self.collision == False:
                    o.move()
                    o.resume_driving()
                    self.reward_sum = 0

            o.draw()
            if o.pos.left > 0:
                inbound_vehicles_driving.append(o)
        self.vehicles_driving = inbound_vehicles_driving

    def _create_boundary(self, car, inner, outer):
        if (car.pos.right <= outer) & (car.pos.left >= inner):
            if car not in self.in_bounds:
                self.in_bounds.append(car)
                self.count += 1
                #xyyyz
                #print("This is the car: " + str(car))
        else:
            if car in self.in_bounds:
                self.in_bounds.remove(car)
                self.count -= 1

    def _create_matrices(self):
        self.position_matrix = np.zeros((1, 7))
        self.velocity_matrix = np.zeros((1, 7))
        for car in self.in_bounds:
            left = car.pos.left
            if (left > 437) & (left < 454):
                self.position_matrix[0][6] = 1
                self.velocity_binary(car, 0, 6)
                # self.velocity(car, 0, 6)
            elif (left > 421) & (left < 438):
                self.position_matrix[0][5] = 1
                self.velocity_binary(car, 0, 5)
                # self.velocity(car, 0, 5)
            elif (left > 405) & (left < 422):
                self.position_matrix[0][4] = 1
                self.velocity_binary(car, 0, 4)
                # self.velocity(car, 0, 4)
            elif (left > 389) & (left < 406):
                self.position_matrix[0][3] = 1
                self.velocity_binary(car, 0, 3)
                # self.velocity(car, 0, 3)
            elif (left > 373) & (left < 390):
                self.position_matrix[0][2] = 1
                self.velocity_binary(car, 0, 2)
                # self.velocity(car, 0, 2)
            elif (left > 357) & (left < 374):
                self.position_matrix[0][1] = 1
                self.velocity_binary(car, 0, 1)
                # self.velocity(car, 0, 1)
            elif (left > 341) & (left < 358):
                self.position_matrix[0][0] = 1
                self.velocity_binary(car, 0, 0)
                # self.velocity(car, 0, 0)

class EastLane(Lane):
    def __init__(self, screen, background):
        Lane.__init__(self, screen, background)

    def generateVehicles(self):
        if np.random.rand() > self.num_of_cars:
            occupy = False
            for _ in self.vehicles_driving:
                if _.pos.left <= 17:
                    occupy = True
                    break
            if occupy == False:
                self.vehicles_driving.append(Car(0, 395, speed=2, screen=self.screen))

    def update(self, light):
        inbound_vehicles_driving = []
        for o in self.vehicles_driving:
            self.blit_background(o.pos, o.pos)
            if ((o.pos.right > 282) & (o.pos.right < 285) & ((light == 1) | (light == 2) | (light == 4))):
                o.idle()
            else:
                self.collision = False
                for e in self.vehicles_driving:
                    if (o.pos.right + o.speed > e.pos.left) & (o.pos.right < e.pos.right):
                        self.collision = True
                        o.idle()
                        break
                if self.collision == False:
                    o.move()
                    o.resume_driving()
                    self.reward_sum = 0

            o.draw()
            if o.pos.left < 622:
                inbound_vehicles_driving.append(o)
        self.vehicles_driving = inbound_vehicles_driving

class NorthLane(Lane):
    def __init__(self, screen, background):
        Lane.__init__(self, screen, background)

    def generateVehicles(self):
        if np.random.rand() > self.num_of_cars:
            occupy = False
            for _ in self.vehicles_driving:
                if _.pos.top >= 729:
                    occupy = True
                    break
            if occupy == False:
                self.vehicles_driving.append(CarVertical(325, 743, speed=-2, screen=self.screen))

    def update(self, light):
        inbound_vehicles_driving = []
        for o in self.vehicles_driving:
            self.blit_background(o.pos, o.pos)
            if (o.pos.bottom > 419) & (o.pos.top < 419) & ((light == 0) | (light == 2) | (light == 4)):
                o.idle()
            else:
                self.collision = False
                for e in self.vehicles_driving:
                    if (o.pos.top + o.speed-5 < e.pos.bottom) & (o.pos.top > e.pos.top):
                        self.collision = True
                        o.idle()
                        break
                if self.collision == False:
                    o.move()
                    o.resume_driving()
                    self.reward_sum = 0
            o.draw()
            if o.pos.bottom > 0:
                inbound_vehicles_driving.append(o)
        self.vehicles_driving = inbound_vehicles_driving

    def _create_boundary(self, car, inner, outer):
        #print(str(car))
        if (car.pos.top >= inner) & (car.pos.bottom <= outer):
            if car not in self.in_bounds:
                self.in_bounds.append(car)
                self.count += 1
                #xyyyz
                #print("This is the car: " + str(car))
        else:
            if car in self.in_bounds:
                self.in_bounds.remove(car)
                self.count -= 1
    def _create_matrices(self):
        self.position_matrix = np.zeros((1, 7))
        self.velocity_matrix = np.zeros((1, 7))
        count = 0
        for car in self.in_bounds:
            pos = car.pos.top
            if (pos > 416) & (pos < 433):
                self.position_matrix[0][0] = 1
                self.velocity_binary(car, 0, 0)
                #self.velocity(car, 0, 0)
            elif (pos > 432) & (pos < 449):
                self.position_matrix[0][1] = 1
                self.velocity_binary(car, 0, 1)
                # self.velocity(car, 0, 1)
            elif (pos > 448) & (pos < 465):
                self.position_matrix[0][2] = 1
                self.velocity_binary(car, 0, 2)
                # self.velocity(car, 0, 2)
            elif (pos > 464) & (pos < 481):
                self.position_matrix[0][3] = 1
                self.velocity_binary(car, 0, 3)
                # self.velocity(car, 0, 3)
            elif (pos > 480) & (pos < 497):
                self.position_matrix[0][4] = 1
                self.velocity_binary(car, 0, 4)
                # self.velocity(car, 0, 4)
            elif (pos > 496) & (pos < 513):
                self.position_matrix[0][5] = 1
                self.velocity_binary(car, 0, 5)
                # self.velocity(car, 0, 5)
            elif (pos > 512) & (pos < 529):
                self.position_matrix[0][6] = 1
                self.velocity_binary(car, 0, 6)
                # self.velocity(car, 0, 6)
            count = count + 1

class SouthLane(Lane):
    def __init__(self, screen, background):
        Lane.__init__(self, screen, background)

    def generateVehicles(self):
        if np.random.rand() > self.num_of_cars:
            occupy = False
            for _ in self.vehicles_driving:
                if _.pos.top <= 15:
                    occupy = True
                    break
            if occupy == False:
                self.vehicles_driving.append(CarVertical(295, 0, speed=2, screen=self.screen))

    def update(self, light):
        inbound_vehicles_driving = []
        for o in self.vehicles_driving:
            self.blit_background(o.pos, o.pos)
            if (o.pos.bottom > 350) & (o.pos.top < 350) & ((light == 0) | (light == 2) | (light == 4)):
                o.idle()
            else:
                self.collision = False
                for e in self.vehicles_driving:
                    if (o.pos.bottom + o.speed+5 > e.pos.top) & (o.pos.bottom < e.pos.bottom):
                        self.collision = True
                        o.idle()
                        break
                if self.collision == False:
                    o.move()
                    o.resume_driving()
                    self.reward_sum = 0
            o.draw()
            if o.pos.top < 744:
                inbound_vehicles_driving.append(o)
        self.vehicles_driving = inbound_vehicles_driving

    def _create_boundary(self, car, inner, outer):
        if (car.pos.top >= outer) & (car.pos.bottom <= inner):
            if car not in self.in_bounds:
                self.in_bounds.append(car)
                self.count += 1
                #xyyyz
                #print("This is the car: " + str(car))
        else:
            if car in self.in_bounds:
                self.in_bounds.remove(car)
                self.count -= 1

    def _create_matrices(self):
        self.position_matrix = np.zeros((1, 7))
        self.velocity_matrix = np.zeros((1, 7))
        count = 0
        for car in self.in_bounds:
            pos = car.pos.bottom
            if (pos > 239) & (pos < 257):
                self.position_matrix[0][6] = 1
                self.velocity_binary(car, 0, 6)
                # self.velocity(car, 0, 6)
            elif (pos > 255) & (pos < 273):
                self.position_matrix[0][5] = 1
                self.velocity_binary(car, 0, 5)
                # self.velocity(car, 0, 5)
            elif (pos > 271) & (pos < 289):
                self.position_matrix[0][4] = 1
                self.velocity_binary(car, 0, 4)
                # self.velocity(car, 0, 4)
            elif (pos > 287) & (pos < 305):
                self.position_matrix[0][3] = 1
                self.velocity_binary(car, 0, 3)
                # self.velocity(car, 0, 3)
            elif (pos > 303) & (pos < 321):
                self.position_matrix[0][2] = 1
                self.velocity_binary(car, 0, 2)
                # self.velocity(car, 0, 2)
            elif (pos > 319) & (pos < 337):
                self.position_matrix[0][1] = 1
                self.velocity_binary(car, 0, 1)
                # self.velocity(car, 0, 1)
            elif (pos > 335) & (pos < 353):
                self.position_matrix[0][0] = 1
                self.velocity_binary(car, 0, 0)
                # self.velocity(car, 0, 0)

            count += 1

class TrafficSignal:
    def __init__(self, screen):
        self.light = 0
        self.red_light = load_image('red_horizontal.jpg').convert()
        self.rv = load_image('red_v.jpg').convert()
        self.green_light = load_image('green_horizontal.jpg').convert()
        self.gv = load_image('green_v.jpg').convert()
        yellow_light = load_image('yellow_traffic_light.png').convert()
        yv = load_image('yellow_traffic_light_v.png').convert()
        self.yellow_light = pygame.transform.scale(yellow_light, (75, 30))
        self.yv = pygame.transform.scale(yv, (30, 75))
        self.screen = screen


    def sequence(self):
        if self.light == 0:
            self.light = 2
        elif self.light == 2:
            self.light = 1
        elif self.light == 1:
            self.light = 4
        elif self.light == 4:
            self.light = 0

    def act(self, action):
        if action == 0:
            self.light = 0
        elif action == 1:
            self.light = 1
        elif action == 2:
            self.light = 2
        elif action == 4:
            self.light = 4

    def currentSignal(self):
        return self.light

    # previously was blit_light
    def draw(self):
        if self.light == 1:
            self.screen.blit(self.red_light, (350, 330))
            self.screen.blit(self.gv, (245, 285))
        elif self.light == 2:
            self.screen.blit(self.yellow_light, (350, 330))
            self.screen.blit(self.rv, (245, 285))
        elif self.light == 0:
            self.screen.blit(self.green_light, (350, 330))
            self.screen.blit(self.rv, (245, 285))
        elif self.light == 4:
            self.screen.blit(self.red_light, (350, 330))
            self.screen.blit(self.yv, (245, 285))

    def step(self):
        #self.east.generateVehicles()
        #self.east.update(self.currentSignal())
        #self.east.draw()
        self.draw()

# TrafficSimulator is the container and controller class. Its constructor makes and embeds
# instances of the car, traffic signal, and lane classes.
class TrafficSim():
    def __init__(self, width=622, height=743):
        self.actions = {
            "left": K_LEFT, # red
            "right": K_RIGHT, #green
            "up": K_UP, #yellow_green
            "down": K_DOWN  #yellow_red
        }
        self.reward_sum = 0.0
        self.collision = False
        self.width = width
        self.height = height
        self._setup()
        self.background = load_image('road-with-lanes2.png').convert()
        self.action = 0
        self.count = 0
        self.num_of_cars = .98
        self.signal_controller = TrafficSignal(self.screen)
        self.north = NorthLane(self.screen, self.background)
        self.south = SouthLane(self.screen, self.background)
        self.east = EastLane(self.screen, self.background)
        self.west = WestLane(self.screen, self.background)
        self.lanes = [self.east, self.west, self.north, self.south]
        self.count = 0
        self.step_index = 0
        #self.position_matrix = np.zeros((16,16))

    def draw(self):
        self.screen.blit(self.background, (0, 0))
        self.signal_controller.draw()

    # def getGameState(self):
    #     observation = np.zeros((16,16))
    def maps(self, lane):
        if(lane == "east"):
            obs = self.east._get_state(284, 163)
            for i in range(7):
                print("This is obs[i] " + str(obs[i]))
                self.position_matrix[8][i] = obs[i]

    def getGameState(self):
        # observation = [(self.east._get_state(284, 163), self.east._get_reward()),
        #     (self.west._get_state(342, 455), self.west._get_reward()),
        #     (self.north._get_state(417, 531), self.north._get_reward()),
        #     ((self.south._get_state(352, 246)), self.south._get_reward())]
        
        observation = np.array([self.east._get_state(284, 163),
            self.west._get_state(342, 455), 
            self.north._get_state(417, 531), 
            self.south._get_state(352, 246)])

        observation = observation.reshape(4,7)
        #print(observation.shape)
        #self.maps("east")
        #observation = self.position_matrix

        reward = self.getReward()
        done = False
        if reward > 20000:
            done = True
        info = self.signal_controller.currentSignal()
        return observation, reward, done, info

    def getReward(self):
        total_reward = 0

        for i in self.lanes:
            total_reward += i._get_reward()

        return total_reward
    
    def game_over(self):
        # not completed
        return self.north.collision | self.south.collision | self.west.collision | self.east.collision
    def init(self):
        self.east.reset()
        self.west.reset()
        self.north.reset()
        self.south.reset()

    def _setup(self):
        """
        Setups up the pygame env, the display and game clock.
        """
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        #self.screen = None
        self.clock = pygame.time.Clock()
        
    def reset(self):
        self.init()
    def _handle_player_events(self):
        for event in pygame.event.get():
            #if event.type in (QUIT, KEYDOWN):
            #    sys.exit()
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key

                if key == self.actions["left"]:
                    self.action = 1

                if key == self.actions["right"]:
                    self.action = 0

                if key == self.actions["up"]:
                    self.action = 2

                if key == self.actions["down"]:
                    self.action = 4
    
    def display_update(self):
        pygame.display.update()
        pygame.time.delay(50)

    def check_for_collisions(self):
        # not completed
        east = self.east.return_intersection()
        north = self.north.return_intersection()
        west = self.west.return_intersection()
        south = self.south.return_intersection()

        for car in east:
            for k in north:
                pass
            for k in south:
                return k.pos.bottom > car.pos.top & k.pos.top < car.pos.bottom & k.pos.right <= car.pos.right & k.pos.left >= car.pos.left
        for car in west:
            for k in north:
                pass
            for j in south:
                pass

    def update_all_lanes(self):
        self.east.generateVehicles()
        self.east.update(self.signal_controller.currentSignal())
        self.west.generateVehicles()
        self.west.update(self.signal_controller.currentSignal())
        self.north.generateVehicles()
        self.north.update(self.signal_controller.currentSignal())
        self.south.generateVehicles()
        self.south.update(self.signal_controller.currentSignal())

    # NEED TO IMPLEMENT THE RENDER METHOD
    def render(self):
        return self.screen

    def step(self, action):
        #dt /= 1000.0
        self.action = action # uncomment in order to make the action work for the gym environment
        self._handle_player_events() # gets which key the player hit
        self.signal_controller.act(self.action)
        self.draw() # draws the background
        self.update_all_lanes()
        self.display_update()
        self.signal_controller.draw()

        state = self.getGameState()
        #print(x)
        #print("Intersection: " + str(self.east.return_intersection()))
        # if self.east._get_reward() > 0 :
        #     print("Reward: " + str(self.east._get_reward()))
        if self.check_for_collisions():
            print("Collision!")
        self.step_index = self.step_index + 1
        #print(state)
        return state

    def getScreenRGB(self):
        """
        Returns the current game screen in RGB format.
        Returns
        --------
        numpy uint8 array
            Returns a numpy array with the shape (width, height, 3).
        """

        return pygame.surfarray.array3d(
            pygame.display.get_surface()).astype(np.uint8)


# quick function to load an image
def load_image(name):
    path = os.path.join(main_dir, 'Traffic_data', name)
    return pygame.image.load(path).convert()

def traffic_signal(red_time, green_time, yellow_time):
    while True:
        for _ in range(green_time):
            yield 0
        for _ in range(yellow_time):
            yield 2
        for _ in range(red_time):
            yield 1
        for _ in range(yellow_time):
            yield 4


"""
if __name__ == '__main__':
    #pygame.init()
    game = TrafficSimulator(width=622, height=743)
    #game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()

    while True:#
        #if game.game_over():
        #    game.init()
        print(game.step_index)
        dt = game.clock.tick_busy_loop(30)
        x = game.step()
        # if game.step_index % 20 == 0:
        #     x = game.step(1)
        # else:
        #     x = game.step(0)
        #pygame.display.update()
        #print(x)

        #time.sleep(.2)
        # """