import pygame
import CGL
import numpy as np
import time

pygame.init()

# Color in32 is of the form: AARRGGBB (alpha, R, G, B).
# If age is selected, you may see "pulsing" this is expect. This is because as the BB counts up in RRGGBB, it resests to zero and adds 1 to G, etc...
def render(cgl, delay, dim, color=0xFF, showAge=False):
    display = pygame.display.set_mode(dim)
    running = True
    side = cgl.get_side() 
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        image = cgl.get_stable().T if showAge else cgl.get_state().T    # Need to transpose to look normal.

        if showAge == False:
            image[image != 0] = color

        #image = image.T.flatten() # Need to transpose image for it to appear properly and flatten for later computations.
        # Commented out -- used to be for int32!
        #if not showAge: # Monochrome just show alive or dead cells, still need to update for new colors.
        #    image = np.where(image, color, 0)
        #    image = image.reshape(side, side)
        #else:
        #    image[image == 0] = background
        #    image
            #image = np.where(image, 0, 0x00FFFFFF) # Low numbers are dark and background is black, inverse so colors show.
        #    image[image == 0] = background
            # Convert 2D matrix of int32 to RGB (R,G,B) tuple for pygame.
            # Traditionally RGB, but formula swapped since Blue always changes.
        #    top = (image >> 16) & 255   # Red  (counts slowest)
        #    mid = (image >> 8) & 255    # Green
        #    base = image & 255          # Blue (counts fastest)
        #else:
        #    image = np.where(image, ~image, 0) # Low numbers are dark and background is black, inverse so colors show.
        #    # Convert 2D matrix of int32 to RGB (R,G,B) tuple for pygame.
        #    # Traditionally RGB, but formula swapped since Blue always changes.
        #    top = (image >> 16) & 255   # Red  (counts slowest)
        #    mid = (image >> 8) & 255    # Green
        #    base = image & 255          # Blue (counts fastest)

        #    image = np.stack((top, mid, base), axis=-1).astype(np.uint8).reshape(side, side, 3) # Need to reshape for RGB values.

        surf = pygame.pixelcopy.make_surface(image)
        surf = pygame.transform.scale(surf, dim)
        pygame.display.set_caption(f"Seed={cgl.get_seed():,} Iteration={cgl.get_count():,} Stability={cgl.sum_stable():,} Alive={cgl.sum_state():,}")
        display.blit(surf, (0, 0))
        pygame.display.update()
        #print(cgl.get_stable(), '\n')
        pygame.time.delay(delay)
        cgl.step()

    pygame.quit()

# 20x20 sample with one oscillator and 1 cell in upper left corner.

"""
sample = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
"""


sample = np.array([[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])

window_height = window_length = 1000
delay_time = 0 # Milliseconds.
sim_side_size = 200
cgl = CGL.sim(side=sim_side_size, seed=1230, gpu=True, spawnStabilityFactor=-20, stableStabilityFactor=20)
#cgl.toggle_state([0,1,2,3,4,5])
#cgl = CGL.sim(state=sample, gpu=True, spawnStabilityFactor=-2, stableStabilityFactor=2)
#cgl.update_state(sample, sample.shape[0])
start = time.perf_counter()
render(cgl, delay=delay_time, dim=(window_height, window_length), showAge=True)
#for _ in range(1000000):
#    print(cgl.get_stable(), '\n')
#    print(cgl.get_state(), '\n')
#    cgl.step()
#    time.sleep(1)
print('Runtime=', time.perf_counter()-start)
