import time

import numpy as np
from retro_contest.local import make

def main():
    env = make(game='SonicTheHedgehog-Genesis', state='SpringYardZone.Act3')
    obs = env.reset()
    while True:
        time.sleep(0.2)
        obs, rew, done, info = env.step(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))
        print(rew)
        # env.render()
        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()