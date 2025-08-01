import pygame
import pandas as pd
import sys
from pathlib import Path
import numpy as np
from resemblyzer import preprocess_wav
import os

def audioplayer(file):
    pygame.mixer.pre_init(16000, size=-16, channels=1, allowedchanges=0)
    pygame.init()

    # file=sys.argv[1]
    assert(os.path.exists('audio/'+Path(file).stem+".wav")) # Verify that an .wav file with same name exists

    screen=pygame.display.set_mode((640,640))
    font=pygame.font.Font(None,size=50)
    smallfont=pygame.font.Font(None,size=25)
    sound=None
    running = True
    clock=pygame.time.Clock()

    wav = preprocess_wav('audio/'+Path(file).stem+".wav")   # Obtain VAD audio so the timestamps sync up
    scaled_wav = np.int16(wav * 32767)
    sound = pygame.sndarray.make_sound(scaled_wav)
    start_playback_time_ticks = pygame.time.get_ticks()
    sound.play()

    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (192, 192, 192) # Silver (a light grey)
    ]
    square_size = 50
    squares = []
    hollowsquares = []

    timestamps=pd.read_csv(file)
    numspeakers=timestamps['speaker'].max()+1
    # numspeakers=10
    text = font.render(f'Number of speakers: {numspeakers}',True,(255,255,255))

    for i in range(numspeakers):
        # Simple static 5 by 2 layout, edit later if it bothers you
        x = 50 + (i % 5) * (square_size + 20)
        y = 50 + (i // 5) * (square_size + 20)
        squares.append(pygame.Rect(x, y, square_size, square_size))
        hollowsquares.append(pygame.Rect(x+1, y+1, square_size-2, square_size-2))

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


        screen.fill((0, 0, 0))
        screen.blit(text,(50,400))
        time=(pygame.time.get_ticks() - start_playback_time_ticks) / 1000.0
        screen.blit(font.render(f'Current Time: {time}',True,(255,255,255)),(50,450))
        screen.blit(smallfont.render(f'Timestamp File: {file}',True,(255,255,255)),(50,500))
        interval=timestamps[(timestamps['start']<time)&(timestamps['end']>time)]
        for i in range(numspeakers):
            pygame.draw.rect(screen, colors[i], squares[i])        
            if i not in interval['speaker'].values:
                pygame.draw.rect(screen, (0,0,0), hollowsquares[i])
        pygame.display.flip() # Refresh screen
        clock.tick(60)  # Delay
        if (time > timestamps['end'].max() + 1): # Added a small buffer
            running = False
    pygame.quit()

if __name__ == "__main__":
    assert(len(sys.argv)>1)
    audioplayer(sys.argv[1])