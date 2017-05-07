#Modified from http://www.pygame.org/project-Very+simple+Pong+game-816-.html
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
import pygame
from pygame.locals import *
import random
import pygame.surfarray as surfarray

miss_count = 0
hit_count = 0

pygame.init()
screen_width = 168
screen_height = 84

bar_width, bar_height = screen_width / 32. / 2, screen_height / 6.
bar_dist_from_edge = screen_width / 64. / 2
circle_diameter = screen_height / 16.
circle_radius = circle_diameter / 2.
bar1_start_x, bar2_start_x = bar_dist_from_edge, screen_width - bar_dist_from_edge
bar_start_y = (screen_height - bar_height) / 2.
bar_max_y = screen_height - bar_height - bar_dist_from_edge
circle_start_x, circle_start_y = (screen_width - circle_diameter) / 2, (screen_height - circle_diameter) / 2.

screen = pygame.display.set_mode((int(screen_width), int(screen_height)), 0, 32)

#Creating 2 bars, a ball and background.
back = pygame.Surface((int(screen_width),int(screen_height)))
background = back.convert()
background.fill((0,0,0))
bar = pygame.Surface((int(bar_width),int(bar_height)))
bar1 = bar.convert()
bar1.fill((255,255,255))
bar2 = bar.convert()
bar2.fill((255,255,255))
circle_surface = pygame.Surface((int(circle_diameter),int(circle_diameter)))
pygame.draw.circle(circle_surface,(255,255,255),(int(circle_radius),int(circle_radius)),int(circle_radius))
circle = circle_surface.convert()
circle.set_colorkey((0,0,0))

# some definitions
bar1_x, bar2_x = bar1_start_x , bar2_start_x
bar1_y, bar2_y = bar_start_y, bar_start_y
circle_x, circle_y = circle_start_x, circle_start_y
bar1_move, bar2_move = 0. , 0.
speed_bar = screen_height * 1.2
bar1_score, bar2_score = 0,0
speed_x = -screen_width / 1.28 / 2
speed_y = random.uniform(-screen_height/1.92, screen_height/1.92)

clock = pygame.time.Clock()
font = pygame.font.SysFont("calibri",8)

def reset():
    global circle_x, circle_y, bar1_y, bar2_y, circle_start_x, circle_start_y, bar_start_y, screen_width, screen_height, speed_x, speed_y
    circle_x, circle_y = circle_start_x, circle_start_y
    speed_x = -screen_width / 1.28 / 2
    speed_y = random.uniform(-screen_height/1.92, screen_height/1.92)

reset()

done = False
while done==False:       
    for event in pygame.event.get(): # User did something
        if event.type == pygame.QUIT: # If user clicked close
            done = True # Flag that we are done so we exit this loop
        if event.type == KEYDOWN:
            if event.key == K_UP or event.key == K_w:
                bar1_move = -ai_speed
            elif event.key == K_DOWN or event.key == K_s:
                bar1_move = ai_speed
        elif event.type == KEYUP:
            if event.key == K_UP or event.key == K_w:
                bar1_move = 0.
            elif event.key == K_DOWN or event.key == K_s:
                bar1_move = 0.
            
    screen.blit(background,(0,0))
    screen.blit(bar1,(bar1_x,bar1_y))
    screen.blit(bar2,(bar2_x,bar2_y))
    screen.blit(circle,(circle_x,circle_y))

    bar1_y += bar1_move
        
    # movement of circle
    time_passed = clock.tick(30)
    time_sec = time_passed / 1000.0
        
    circle_x += speed_x * time_sec
    circle_y += speed_y * time_sec
    ai_speed = speed_bar * time_sec
    
    # AI of the computer.
    if circle_x >= screen_width / 2:
        if not bar2_y == circle_y + circle_radius:
            bar2_y += (circle_y - bar2_y) / 2

    # keep bars in bounds
    if bar1_y >= bar_max_y: bar1_y = bar_max_y
    elif bar1_y <= bar_dist_from_edge: bar1_y = bar_dist_from_edge
    if bar2_y >= bar_max_y: bar2_y = bar_max_y
    elif bar2_y <= bar_dist_from_edge: bar2_y = bar_dist_from_edge

    # ball hits left bar
    if circle_x <= bar1_x + bar_width:
        if circle_y >= (bar1_y - circle_radius) and circle_y <= (bar1_y + circle_radius + bar_height):
            circle_x = bar_dist_from_edge + bar_width
            speed_x = -speed_x
            hit_count += 1
                
    # ball hits right bar
    if circle_x >= bar2_x - bar_width:
        if circle_y >= (bar2_y - circle_radius) and circle_y <= (bar2_y + circle_radius + bar_height):
            circle_x = screen_width - bar_width
            speed_x = -speed_x


    # bar 1 loses
    if circle_x < -circle_radius:
        bar2_score += 1
        miss_count += 1
        reset()
    # bar 2 loses
    elif circle_x > screen_width + circle_radius:
        bar1_score += 1
        reset()

    # ball hits bottom
    if circle_y <= circle_radius:
        speed_y = -speed_y
        circle_y = circle_radius
    # ball hits top
    elif circle_y >= screen_height - circle_radius:
        speed_y = -speed_y
        circle_y = screen_height - circle_radius

    pygame.display.update()
            
pygame.quit()

