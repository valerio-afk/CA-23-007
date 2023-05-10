import pygame
import numpy as np
from multiprocessing import cpu_count
from typing import List,Tuple,Union
from joblib import Parallel,delayed


#This class represents a single cell pattern to be placed with the mouse
class CellPattern:
    def __init__(this,pattern:Union[np.ndarray,None]=None):
        this.__pattern = np.asarray([[1]]) if pattern is None else pattern
        this.__flip = False

    #this property returns the pattern itself
    @property
    def get_pattern(this):
        return this.__pattern if this.flip == False else this.__pattern[-1::-1,:]

    #this property is to horizontally flip the patterns
    @property
    def flip(this):
        return this.__flip

    @flip.setter
    def flip(this,v:bool):
        this.__flip = v

    #this property allows to centre the patterns wrt the mouse cursor
    @property
    def get_offset(this):
        w,h = this.get_pattern.shape

        w = max(w//2-1,0)
        h = max(h//2-1,0)

        return (w,h)

#a 2x2 square pattern
class SquarePattern (CellPattern):
    def __init__(this):
        super().__init__(np.asarray([[1,1],[1,1]]))

#Copperhead pattern definition
class Copperhead(CellPattern):
    def __init__(this):
        pattern = np.zeros((12,8))
        square = SquarePattern()

        pattern[0:2,3:5] = square.get_pattern
        pattern[9:11, 3:5] = square.get_pattern

        pattern[3,2:6] = 1
        pattern[4, 1:3] = 1
        pattern[4, 5:7] = 1
        pattern[5,0] = 1
        pattern[5,7] = 1

        pattern[7:9, 0] = 1
        pattern[7:9, 7] = 1

        pattern[8, 2] = 1
        pattern[8, 5] = 1

        pattern[-1, 1:3] = 1
        pattern[-1, 5:7] = 1

        super().__init__(pattern)

#Glider gun pattern
class GosperGliderGun(CellPattern):
    def __init__(this):
        pattern = np.zeros((36,9))
        square = SquarePattern()

        pattern[0:2,4:6] = square.get_pattern
        pattern[-2:, 2:4] = square.get_pattern

        pattern[10, 4:7] = 1
        pattern[16, 4:7] = 1
        pattern[11, 3] = 1
        pattern[11, 7] = 1
        pattern[12:14, 2] = 1
        pattern[12:14, 8] = 1

        pattern[14, 5] = 1
        pattern[17, 5] = 1
        pattern[15, 3] = 1
        pattern[15, 7] = 1

        pattern[20:22,2:5]=1
        pattern[22, 1] = 1
        pattern[22, 5] = 1
        pattern[24, 0:2] = 1
        pattern[24, 5:7] = 1

        super().__init__(pattern)

class Glider(CellPattern):
    def __init__(this):
        pattern = np.zeros((3,3))
        pattern[:,-1] = 1
        pattern[-1,1:] = 1
        pattern[1,0] = 1

        super().__init__(pattern)

#source: https://www.geeksforgeeks.org/gcd-in-python/
#this function computes the gcd that is then used to calculate the cell proportions
def greatest_common_divisor(a:int,b:int) -> int:
    if(b == 0):
        return abs(a)
    else:
        return greatest_common_divisor(b, a % b)

#this function calculates the available sized of squares to determine the zoom levels
def available_sizes(num:int)->List[int]:
    result = []

    while ((num%2)==0):
        result.append(num)
        num //=2

    return sorted(result)


#display the game of life main grid on screen
def display_grid(surface:pygame.Surface, size:int) -> None:
    (w,h) = surface.get_size()

    #draw vertical lines:
    for x in range(size,w,size):
        pygame.draw.line(surface, "black", (x,0),(x,h),1)

    # draw horizontal lines:
    for y in range(size, h, size):
        pygame.draw.line(surface, "black", (0,y), (w, y), 1)

#display a single cell on screen
def display_cell(surface:pygame.Surface, x:int,y:int, size:int, color) ->None:
    cell = pygame.Rect((x * size+1, y * size+1), (size-1, size-1))
    pygame.draw.rect(surface, color, cell, 0)

#display a set (nd array) of cells on screen
def display_cells(surface:pygame.Surface, world:np.ndarray, size:int, color) -> None:
    x_coords,y_coords = np.nonzero(world)

    for x,y in zip(x_coords, y_coords):
        display_cell(surface,x,y,size,color)

#display the cells currently evolving
def display_world(surface:pygame.Surface, world:np.ndarray, size:int) -> None:
    display_cells(surface,world,size,"black")

#display cells that the user has placed ready to go
def display_new_world(surface:pygame.Surface, world:np.ndarray, size:int) -> None:
    display_cells(surface,world,size,"green")

#this function servers to check if a cell is alive
is_alive = lambda c : c!=0

#this is the main function of the game. It makes the world moving
def evolve_game(world:np.ndarray) -> np.ndarray:
    #A new world with no alive cells is generated
    next_world = np.zeros(world.shape)
    #get the size of it
    w,h = world.shape


    #for each pair of coordinates
    for x in range(0,w):
        for y in range(0,h):
            #get 8-neighbours and make sure they are wihtin the bounds of the world
            a = max(x-1,0)
            b = min(x + 1, w)
            c = max(y - 1, 0)
            d = min(y + 1, h)

            #get the neighbours of the current cell
            neighbours = world[a:b+1,c:d+1]
            #sum the number of alive cells
            n=neighbours.sum()

            #apply rules!
            if ((is_alive(world[x,y]) and (2<=(n-1)<=3)) or #the -1 is important because the current cell doesn't count!!!!
                (not is_alive(world[x,y]) and (n==3))):
                next_world[x,y] = 1

    return next_world #return the new configuration

def evolve_parallel(world:np.ndarray,n_cores:int) -> np.ndarray:
    MARGIN = 5
    c,r = None,None

    World = np.pad(world,MARGIN)
    new_World = np.zeros(World.shape)

    while ((c,r)==(None,None)) and (n_cores>1):
        divisors = calculate_prime_factors(n_cores)
        ndiv = len(divisors)
        if (ndiv>=2):
            half = round(ndiv/2)

            c = 1
            for _ in range(half):
                d,n = divisors.popitem()
                c *= d**n

            r = 1
            for _ in range(ndiv-half):
                d, n = divisors.popitem()
                r *= d ** n
        elif ndiv==1:
            d, n = divisors.popitem()
            if (n>=2):
                r = d**(n//2)
                c = d**(n-(n//2))

        n_cores-=1

    if (c == None): c = 1
    if (r == None): r = 1

    w,h = world.shape

    block_width = w // c
    block_height= h // r

    parallel = Parallel(n_jobs=n_cores)
    delayed_functions = [None] * (r*c)

    i = 0

    for rr in range(r):
        for cc in range(c):
            x1 = cc * block_width
            x2 = ((cc+1) * block_width) + (MARGIN * 2)

            y1 = rr * block_height
            y2 = ((rr + 1) * block_height) + (MARGIN * 2)

            delayed_functions[i] = delayed(evolve_game)(World[x1:x2,y1:y2])

            i+=1

    blocks = parallel(delayed_functions)
    i=0

    for rr in range(r):
        for cc in range(c):

            x1 = cc * block_width
            x2 = ((cc+1) * block_width) + (MARGIN * 2)

            y1 = rr * block_height
            y2 = ((rr + 1) * block_height) + (MARGIN * 2)

            new_World[x1+MARGIN:x2-MARGIN,y1+MARGIN:y2-MARGIN] = blocks[i][MARGIN:-MARGIN,MARGIN:-MARGIN]
            i+=1

    return new_World[MARGIN:-MARGIN,MARGIN:-MARGIN]




#taken from https://www.pythonforbeginners.com/basics/find-prime-factors-of-a-number-in-python
def calculate_prime_factors(N):
    prime_factors = {}
    if N % 2 == 0:
        prime_factors[2] = 0
    while N % 2 == 0:
        N = N // 2
        prime_factors[2]+=1
        if N == 1:
            return prime_factors
    for factor in range(3, N + 1, 2):
        if N % factor == 0:
            prime_factors[factor] = 0
            while N % factor == 0:
                N = N // factor
                prime_factors[factor]+=1
                if N == 1:
                    return prime_factors

#this is an helper function that prints the current position of the mouse
def display_mouse_cell(surface:pygame.Surface, pos:Tuple[int,int],size:int,pattern:CellPattern) -> None:
    u,v = pos
    u //= size
    v //= size

    #w,h = surface.get_size()
    x,y = np.nonzero(pattern.get_pattern)

    dx,dy = pattern.get_offset

    for xx,yy in zip (x,y):
        X = xx+u - dx
        Y = yy+v - dy
        display_cell(surface,X,Y,size,"pink")

def init_world(window_size,cell_size:int) -> Tuple[np.ndarray,np.ndarray]:
    world = np.zeros((window_size[0] // cell_size, window_size[1] // cell_size))
    new_world = np.zeros(world.shape)

    return world,new_world

pygame.display.init()
info = pygame.display.Info()

window_size = (info.current_w,info.current_h)

PARALLEL = True

#calculate the gcd with the window size values
gcd = greatest_common_divisor(*window_size)
#calculate the number of available sizes given the gcd
sizes = available_sizes(gcd)
#default zoom level
zoom=3
#get default cell size
size = sizes[zoom]
update_time=500 #update after 500ms (default)

#initalise the new world with the default configuration
world,new_world = init_world(window_size,size)

#default mouse pattern
mouse_pattern = CellPattern()

#initialise the game
pygame.init()

#setup the game environment
window = pygame.display.set_mode(window_size)
pygame.display.set_caption("AFK :: Game of Life <afk.broadcast@gmail.com>")
clock = pygame.time.Clock()
running = True
grid = True #display grid

#parallelism
n_cores = cpu_count() - 1 #take one out for the main thread
n_cores -= n_cores%2 #consider only even numbers

#delta time (dt) is necessary to define the update rate which is different than the frame rate
dt = 0

while running:
    # poll for events
    for event in pygame.event.get():
        # pygame.QUIT event means the user clicked X to close your window
        if event.type == pygame.QUIT:
            running = False
        #a key has been pressed and released
        elif event.type == pygame.KEYUP:
            #if "r" is pressed, initialise the world with a random seed
            if event.key == pygame.K_r:
                new_world = np.random.rand(window_size[0]//size,(window_size[1])//size)>0.75
            #if "k" is pressed, kill all cells
            elif event.key == pygame.K_k:
                world,new_world = init_world(window_size,size)
            #if "+" in the keypad is pressed, speed up the evolution time
            elif event.key == pygame.K_KP_PLUS:
                update_time-=10
                if (update_time<20):
                    update_time=20
            # if "-" in the keypad is pressed, slow down the evolution time
            elif event.key == pygame.K_KP_MINUS:
                update_time+=10
                if (update_time<1000):
                    update_time=1000
            #if "z" is pressed, zoom out the world
            elif event.key == pygame.K_z:
                zoom = max(zoom-1,0)
                size = sizes[zoom]
                world,new_world = init_world(window_size,size)
            # if "x" is pressed, zoom in the world
            elif event.key == pygame.K_x:
                zoom = min(zoom+1,len(sizes)-1)
                size = sizes[zoom]
                world,new_world = init_world(window_size,size)
            # if "f" is pressed, horizontal flip the current pattern
            elif event.key == pygame.K_f:
                mouse_pattern.flip = not mouse_pattern.flip
            # pattern 1: single cell
            elif (event.key == pygame.K_1) or (event.key == pygame.K_KP1):
                mouse_pattern = CellPattern()
            # pattern 2: square pattern
            elif (event.key == pygame.K_2) or (event.key == pygame.K_KP2):
                mouse_pattern = SquarePattern()
            # pattern 3: Copperhead Pattern
            elif (event.key == pygame.K_3) or (event.key == pygame.K_KP3):
                mouse_pattern = Copperhead()
            # pattern 4: Gosper Glider Gun
            elif (event.key == pygame.K_4) or (event.key == pygame.K_KP4):
                mouse_pattern = GosperGliderGun()
            elif (event.key == pygame.K_5) or (event.key == pygame.K_KP5):
                mouse_pattern = Glider()
            #Q key pressed to quit the game
            elif(event.key == pygame.K_q):
                running = False
            elif (event.key == pygame.K_g):
                grid = not grid
        #a mouse button is pressed and released
        elif event.type == pygame.MOUSEBUTTONUP:
            #if left button is pressed, place new cells in the new world to be generated
            if (event.button==1):
                u, v = event.pos
                u //= size
                v //= size

                x, y = np.nonzero(mouse_pattern.get_pattern)
                w,h = new_world.shape
                dx, dy = mouse_pattern.get_offset

                for xx, yy in zip(x, y):
                    X = xx + u - dx
                    Y = yy + v - dy

                    if ((0<=X<w) and (0<=Y<h)):
                        new_world[X,Y] = 1-new_world[X,Y]
            #if right button is pressed, the new world starts and merges with the current one.
            elif (event.button==3):
                world[np.nonzero(new_world)] = 1
                _,new_world = init_world(window_size,size)




    # fill the screen with a color to wipe away anything from last frame
    window.fill("white")

    # print cell grid
    if grid:
        display_grid(window,size)
    # display current cell configuration
    display_world(window,world,size)

    #if mouse is on the window, print current pattern
    if (pygame.mouse.get_focused()):
        display_mouse_cell(window,pygame.mouse.get_pos(),size,mouse_pattern)

    #display green cells, ie the ones that the user wants to add up
    display_new_world(window,new_world,size)


    # flip() the display to put your work on screen
    pygame.display.flip()

    #wait and get the time in ms
    dt += clock.tick(60)

    #if the dt is => the update time, then evolve the world
    if ((dt>=update_time) and (world.sum()>0)):
        if (PARALLEL):
            world = evolve_parallel(world, n_cores)
        else:
            world = evolve_game(world)

        dt=0


pygame.quit()