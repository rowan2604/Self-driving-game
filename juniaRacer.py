import pygame
import random
import os
import math
import numpy as np
import time
import sys
from datetime import datetime
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from PIL import Image
from operator import attrgetter
import importlib

driver = importlib.import_module("drivers.c3po", package = None)

pygame.init() #Initialize pygame
#Some variables initializations
img = 0 #This one is used when recording frames
size = width,height = 1600, 900 #Size to use when creating pygame window

#Colors
white = (255,255,255)
green = (0, 255, 0) 
blue = (0, 0, 128)  
black = (0,0,0)
gray = pygame.Color('gray12')
Color_line = (255,0,0)

generation = 1
mutationRate = 90
FPS = 90
#selectedCars = []
selected = 0
lines = True #If true then lines of player are shown
player = True #If true then player is shown
display_info = True #If true then display info is shown
frames = 0
maxspeed = 10 
number_track = 1

white_small_car = pygame.image.load('Images\Sprites\white_small.png')
white_big_car = pygame.image.load('Images\Sprites\white_big.png')
green_small_car = pygame.image.load('Images\Sprites\green_small.png')
green_big_car = pygame.image.load('Images\Sprites\green_big.png')

bg = pygame.image.load('bg73.png')
bg4 = pygame.image.load('bg43.png')


def calculateDistance(x1,y1,x2,y2): #Used to calculate distance between points
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def rotation(origin, point, angle): #Used to rotate points #rotate(origin, point, math.radians(10))
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy
    
def move(point, angle, unit): #Translate a point in a given direction
  x = point[0]
  y = point[1]
  rad = math.radians(-angle % 360)

  x += unit*math.sin(rad)
  y += unit*math.cos(rad)

  return x, y
  
def sigmoid(z): #Sigmoid function, used as the neurons activation function
    return 1.0/(1.0+np.exp(-z))

class Cell:
    #A wall separates a pair of cells in the N-S or W-E directions.
    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

    def __init__(self, x, y):
        self.x, self.y = x, y
        self.walls = {'N': True, 'S': True, 'E': True, 'W': True}
        self.color = 0, 0, 0
        self.track = ""

    def has_all_walls(self):
        return all(self.walls.values())

    def knock_down_wall(self, other, wall):
        #Knock down the wall between cells self and other
        self.walls[wall] = False
        other.walls[Cell.wall_pairs[wall]] = False

class Car:
  def __init__(self, sizes):
    self.score = 0
    self.num_layers = len(sizes) #Number of nn layers
    self.sizes = sizes #List with number of neurons per layer
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #Biases
    self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] #Weights 
    #c1, c2, c3, c4, c5 are five 2D points where the car could collided, updated in every frame
    self.c1 = 0,0
    self.c2 = 0,0
    self.c3 = 0,0
    self.c4 = 0,0
    self.c5 = 0,0
    #d1, d2, d3, d4, d5 are distances from the car to those points, updated every frame too and used as the input for the NN
    self.d1 = 0
    self.d2 = 0
    self.d3 = 0
    self.d4 = 0
    self.d5 = 0
    self.yaReste = False
    #The input and output of the NN must be in a numpy array format
    self.inp = np.array([[self.d1],[self.d2],[self.d3],[self.d4],[self.d5]])
    self.outp = np.array([[0],[0],[0],[0]])
    #Boolean used for toggling distance lines
    self.showlines = True
    #Initial location of the car
    self.x = 120
    self.y = 480
    self.center = self.x, self.y
    #Height and width of the car
    self.height = 35 #45
    self.width = 17 #25
    #These are the four corners of the car, using polygon instead of rectangle object, when rotating or moving the car, we rotate or move these
    self.d = self.x-(self.width/2),self.y-(self.height/2)
    self.c = self.x + self.width-(self.width/2), self.y-(self.height/2)
    self.b = self.x + self.width-(self.width/2), self.y + self.height-(self.height/2) #El rectangulo está centrado en (x,y)
    self.a = self.x-(self.width/2), self.y + self.height-(self.height/2)              #(a), (b), (c), (d) son los vertices
    #Velocity, acceleration and direction of the car
    self.velocity = 0
    self.acceleration = 0  
    self.angle = 180
    #Boolean which goes true when car collides
    self.collided = False
    #Car color and image
    self.color = white
    self.car_image = white_small_car
  def set_accel(self, accel): 
    self.acceleration = accel
  def rotate(self, rot): 
    self.angle += rot
    if self.angle > 360:
        self.angle = 0
    if self.angle < 0:
        self.angle = 360 + self.angle
  def update(self): #En cada frame actualizo los vertices (traslacion y rotacion) y los puntos de colision
    self.score += self.velocity
    if self.acceleration != 0:
        self.velocity += self.acceleration
        if self.velocity > maxspeed:
            self.velocity = maxspeed
        elif self.velocity < 0:
            self.velocity = 0
    else:
        self.velocity *= 0.92
        
    self.x, self.y = move((self.x, self.y), self.angle, self.velocity)
    self.center = self.x, self.y
    
    self.d = self.x-(self.width/2),self.y-(self.height/2)
    self.c = self.x + self.width-(self.width/2), self.y-(self.height/2)
    self.b = self.x + self.width-(self.width/2), self.y + self.height-(self.height/2) #El rectangulo está centrado en (x,y)
    self.a = self.x-(self.width/2), self.y + self.height-(self.height/2)              #(a), (b), (c), (d) son los vertices
        
    self.a = rotation((self.x,self.y), self.a, math.radians(self.angle)) 
    self.b = rotation((self.x,self.y), self.b, math.radians(self.angle))  
    self.c = rotation((self.x,self.y), self.c, math.radians(self.angle))  
    self.d = rotation((self.x,self.y), self.d, math.radians(self.angle))

    self.c1 = move((self.x,self.y),self.angle,10)
    while bg4.get_at((int(self.c1[0]),int(self.c1[1]))).a!=0:
        self.c1 = move((self.c1[0],self.c1[1]),self.angle,10)
    while bg4.get_at((int(self.c1[0]),int(self.c1[1]))).a==0:
        self.c1 = move((self.c1[0],self.c1[1]),self.angle,-1)

    self.c2 = move((self.x,self.y),self.angle+45,10)
    while bg4.get_at((int(self.c2[0]),int(self.c2[1]))).a!=0:
        self.c2 = move((self.c2[0],self.c2[1]),self.angle+45,10)
    while bg4.get_at((int(self.c2[0]),int(self.c2[1]))).a==0:
        self.c2 = move((self.c2[0],self.c2[1]),self.angle+45,-1)

    self.c3 = move((self.x,self.y),self.angle-45,10)
    while bg4.get_at((int(self.c3[0]),int(self.c3[1]))).a!=0:
        self.c3 = move((self.c3[0],self.c3[1]),self.angle-45,10)
    while bg4.get_at((int(self.c3[0]),int(self.c3[1]))).a==0:
        self.c3 = move((self.c3[0],self.c3[1]),self.angle-45,-1)

    self.c4 = move((self.x,self.y),self.angle+90,10)
    while bg4.get_at((int(self.c4[0]),int(self.c4[1]))).a!=0:
        self.c4 = move((self.c4[0],self.c4[1]),self.angle+90,10)
    while bg4.get_at((int(self.c4[0]),int(self.c4[1]))).a==0:
        self.c4 = move((self.c4[0],self.c4[1]),self.angle+90,-1)

    self.c5 = move((self.x,self.y),self.angle-90,10)
    while bg4.get_at((int(self.c5[0]),int(self.c5[1]))).a!=0:
        self.c5 = move((self.c5[0],self.c5[1]),self.angle-90,10)
    while bg4.get_at((int(self.c5[0]),int(self.c5[1]))).a==0:
        self.c5 = move((self.c5[0],self.c5[1]),self.angle-90,-1)
        
    self.d1 = int(calculateDistance(self.center[0], self.center[1], self.c1[0], self.c1[1]))
    self.d2 = int(calculateDistance(self.center[0], self.center[1], self.c2[0], self.c2[1]))
    self.d3 = int(calculateDistance(self.center[0], self.center[1], self.c3[0], self.c3[1]))
    self.d4 = int(calculateDistance(self.center[0], self.center[1], self.c4[0], self.c4[1]))
    self.d5 = int(calculateDistance(self.center[0], self.center[1], self.c5[0], self.c5[1]))
    
    
  def draw(self,display):
    rotated_image = pygame.transform.rotate(self.car_image, -self.angle-180)
    rect_rotated_image = rotated_image.get_rect()
    rect_rotated_image.center = self.x, self.y
    gameDisplay.blit(rotated_image, rect_rotated_image)
  
    center = self.x, self.y
    if self.showlines: 
        pygame.draw.line(gameDisplay,Color_line,(self.x,self.y),self.c1,2)
        pygame.draw.line(gameDisplay,Color_line,(self.x,self.y),self.c2,2)
        pygame.draw.line(gameDisplay,Color_line,(self.x,self.y),self.c3,2)
        pygame.draw.line(gameDisplay,Color_line,(self.x,self.y),self.c4,2)
        pygame.draw.line(gameDisplay,Color_line,(self.x,self.y),self.c5,2) 
    
  def showLines(self):
    self.showlines = not self.showlines
    
  def collision(self):
      if (bg4.get_at((int(self.a[0]),int(self.a[1]))).a==0) or (bg4.get_at((int(self.b[0]),int(self.b[1]))).a==0) or (bg4.get_at((int(self.c[0]),int(self.c[1]))).a==0) or (bg4.get_at((int(self.d[0]),int(self.d[1]))).a==0):
        return True
      else:
        return False    

  def resetPosition(self):
      self.x = 120
      self.y = 480
      self.angle = 180
      return
      
  def takeAction(self): 
    if self.outp.item(0) > 0.5: #Accelerate
        self.set_accel(0.2)
    else:
        self.set_accel(0)      
    if self.outp.item(1) > 0.5: #Brake
        self.set_accel(-0.2)     
    if self.outp.item(2) > 0.5: #Turn right
        self.rotate(-5)    
    if self.outp.item(3) > 0.5: #Turn left
        self.rotate(5) 
    return
  

#These is just the text being displayed on pygame window
infoX = 1365
infoY = 600 
font = pygame.font.Font('freesansbold.ttf', 18) 
text1 = font.render('0..9 - Change Mutation', True, white) 
text2 = font.render('LMB - Select/Unselect', True, white)
text3 = font.render('RMB - Delete', True, white)
text4 = font.render('L - Show/Hide Lines', True, white)
text5 = font.render('R - Reset', True, white)
text6 = font.render('B - Breed', True, white)
text7 = font.render('C - Clean', True, white)
text8 = font.render('N - Next Track', True, white)
text9 = font.render('A - Toggle Player', True, white)
text10 = font.render('D - Toggle Info', True, white)
text11 = font.render('M - Breed and Next Track', True, white)
text1Rect = text1.get_rect().move(infoX,infoY)
text2Rect = text2.get_rect().move(infoX,infoY+text1Rect.height)
text3Rect = text3.get_rect().move(infoX,infoY+2*text1Rect.height)
text4Rect = text4.get_rect().move(infoX,infoY+3*text1Rect.height)
text5Rect = text5.get_rect().move(infoX,infoY+4*text1Rect.height)
text6Rect = text6.get_rect().move(infoX,infoY+5*text1Rect.height)
text7Rect = text7.get_rect().move(infoX,infoY+6*text1Rect.height)
text8Rect = text8.get_rect().move(infoX,infoY+7*text1Rect.height)
text9Rect = text9.get_rect().move(infoX,infoY+8*text1Rect.height)
text10Rect = text10.get_rect().move(infoX,infoY+9*text1Rect.height)
text11Rect = text11.get_rect().move(infoX,infoY+10*text1Rect.height)

def displayTexts():  
    infotextX = 20
    infotextY = 600
    infotext1 = font.render('Gen ' + str(generation), True, white) 
    #infotext2 = font.render('Cars: ' + str(num_of_nnCars), True, white)
    infotext3 = font.render('Alive: ' + str(alive), True, white)
    infotext4 = font.render('Selected: ' + str(selected), True, white)
    if lines == True:
        infotext5 = font.render('Lines ON', True, white)
    else:
        infotext5 = font.render('Lines OFF', True, white)
    if player == True:
        infotext6 = font.render('Player ON', True, white)
    else:
        infotext6 = font.render('Player OFF', True, white)
    #infotext7 = font.render('Mutation: '+ str(2*mutationRate), True, white)
    #infotext8 = font.render('Frames: ' + str(frames), True, white)
    infotext9 = font.render('FPS: 30', True, white)
    infotext1Rect = infotext1.get_rect().move(infotextX,infotextY)
    infotext2Rect = infotext2.get_rect().move(infotextX,infotextY+infotext1Rect.height)
    infotext3Rect = infotext3.get_rect().move(infotextX,infotextY+2*infotext1Rect.height)
    infotext4Rect = infotext4.get_rect().move(infotextX,infotextY+3*infotext1Rect.height)
    infotext5Rect = infotext5.get_rect().move(infotextX,infotextY+4*infotext1Rect.height)
    infotext6Rect = infotext6.get_rect().move(infotextX,infotextY+5*infotext1Rect.height)
    #infotext7Rect = infotext7.get_rect().move(infotextX,infotextY+6*infotext1Rect.height)
    #infotext8Rect = infotext8.get_rect().move(infotextX,infotextY+7*infotext1Rect.height)
    infotext9Rect = infotext9.get_rect().move(infotextX,infotextY+6*infotext1Rect.height)

    gameDisplay.blit(text1, text1Rect)  
    gameDisplay.blit(text2, text2Rect)  
    gameDisplay.blit(text3, text3Rect) 
    gameDisplay.blit(text4, text4Rect) 
    gameDisplay.blit(text5, text5Rect) 
    gameDisplay.blit(text6, text6Rect)
    gameDisplay.blit(text7, text7Rect)   
    gameDisplay.blit(text8, text8Rect)  
    gameDisplay.blit(text9, text9Rect)     
    gameDisplay.blit(text10, text10Rect) 
    gameDisplay.blit(text11, text11Rect)  
    
    gameDisplay.blit(infotext1, infotext1Rect)  
    gameDisplay.blit(infotext2, infotext2Rect)  
    gameDisplay.blit(infotext3, infotext3Rect) 
    gameDisplay.blit(infotext4, infotext4Rect) 
    gameDisplay.blit(infotext5, infotext5Rect) 
    gameDisplay.blit(infotext6, infotext6Rect)
    gameDisplay.blit(infotext9, infotext9Rect) 
    return
 

gameDisplay = pygame.display.set_mode(size) #creates screen
clock = pygame.time.Clock()

inputLayer = 6
hiddenLayer = 6
outputLayer = 4
car = Car([inputLayer, hiddenLayer, outputLayer])
auxcar = Car([inputLayer, hiddenLayer, outputLayer])

   
def redrawGameWindow(): #Called on very frame   

    global alive  
    global frames
    global img
    
    frames += 1

    gameD = gameDisplay.blit(bg, (0,0))  
       
    #Same but for player
    if player:
        car.update()
        if car.collision():
            print ("Collision after ")
            car.resetPosition()
            car.update()
        car.draw(gameDisplay)    
    #if display_info:    
        #displayTexts() 
    pygame.display.update() #updates the screen
    #Take a screenshot of every frame
    #pygame.image.save(gameDisplay, "pygameVideo/screenshot" + str(img) + ".jpeg")
    #img += 1

driver.setup()
    
while True:
    #now1 = time.time()  
  
    for event in pygame.event.get(): #Check for events
        if event.type == pygame.QUIT:
            pygame.quit() #quits
            quit()
            
        if event.type == pygame.MOUSEBUTTONDOWN:
            #This returns a tuple:
            #(leftclick, middleclick, rightclick)
            #Each one is a boolean integer representing button up/down.
            mouses = pygame.mouse.get_pressed()
            if mouses[0]:
                pos = pygame.mouse.get_pos()
                point = Point(pos[0], pos[1])

            if mouses[2]:
                pos = pygame.mouse.get_pos()
                point = Point(pos[0], pos[1])

    driver_action = driver.drive(car.d1/1000, car.d2/1000, car.d3/1000, car.d4/1000, car.d5, car.velocity/10, car.acceleration)
    print(driver_action)
    # You can override agent action by keyboard
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        driver_action = driver.LEFT5
    elif keys[pygame.K_RIGHT]:
        driver_action = driver.RIGHT5
    elif keys[pygame.K_UP]:
        driver_action = driver.ACCELERATE
    elif keys[pygame.K_DOWN]:
        driver_action = driver.BRAKE
    # else :
    #     driver_action = driver.NOTHING


    print ("ACTION :" , driver_action, car.d1, car.d2, car.d3, car.d4, car.d5, car.velocity, car.acceleration)


    if driver_action == driver.LEFT5 :
        car.rotate(-5)
    elif driver_action == driver.RIGHT5 :
        car.rotate(5)
    elif driver_action == driver.ACCELERATE :
        car.set_accel(0.2)
    elif driver_action == driver.BRAKE :
        car.set_accel(-0.2)
    # else: # NOTHING
    #     car.set_accel(0)

    redrawGameWindow()

    clock.tick(FPS)




    