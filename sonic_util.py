import numpy as np

GAMES = ['SonicTheHedgehog-Genesis', 'SonicTheHedgehog2-Genesis', 'SonicAndKnuckles3-Genesis']
LEVELS1 = ['SpringYardZone.Act3', 'SpringYardZone.Act2', 'GreenHillZone.Act3', 'GreenHillZone.Act1',
          'StarLightZone.Act2', 'StarLightZone.Act1', 'MarbleZone.Act2', 'MarbleZone.Act1', 'MarbleZone.Act3',
          'ScrapBrainZone.Act2', 'LabyrinthZone.Act2', 'LabyrinthZone.Act1', 'LabyrinthZone.Act3',
          'SpringYardZone.Act1', 'GreenHillZone.Act2', 'StarLightZone.Act3', 'ScrapBrainZone.Act1']
LEVELS2 = ['EmeraldHillZone.Act1', 'EmeraldHillZone.Act2', 'ChemicalPlantZone.Act2', 'ChemicalPlantZone.Act1',
          'MetropolisZone.Act1', 'MetropolisZone.Act2', 'OilOceanZone.Act1', 'OilOceanZone.Act2',
          'MysticCaveZone.Act2', 'MysticCaveZone.Act1', 'HillTopZone.Act1', 'CasinoNightZone.Act1',
          'WingFortressZone', 'AquaticRuinZone.Act2', 'AquaticRuinZone.Act1',
          'MetropolisZone.Act3', 'HillTopZone.Act2', 'CasinoNightZone.Act2']
LEVELS3 = ['LavaReefZone.Act2', 'CarnivalNightZone.Act2', 'CarnivalNightZone.Act1', 'MarbleGardenZone.Act1',
          'MarbleGardenZone.Act2', 'MushroomHillZone.Act2', 'MushroomHillZone.Act1', 'DeathEggZone.Act1',
          'DeathEggZone.Act2', 'FlyingBatteryZone.Act1', 'SandopolisZone.Act1', 'SandopolisZone.Act2',
          'HiddenPalaceZone', 'HydrocityZone.Act2', 'IcecapZone.Act1', 'IcecapZone.Act2', 'AngelIslandZone.Act1',
          'LaunchBaseZone.Act2', 'LaunchBaseZone.Act1', 'LavaReefZone.Act1',
          'FlyingBatteryZone.Act2', 'HydrocityZone.Act1', 'AngelIslandZone.Act2']

TEST_SET = ['SpringYardZone.Act1', 'GreenHillZone.Act2', 'StarLightZone.Act3', 'ScrapBrainZone.Act1', 
            'MetropolisZone.Act3', 'HillTopZone.Act2', 'CasinoNightZone.Act2', 'LavaReefZone.Act1', 
            'FlyingBatteryZone.Act2', 'HydrocityZone.Act1', 'AngelIslandZone.Act2']

def get_sonic_specific_actions():
    buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
    actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                ['DOWN', 'B'], ['B'], [], ['LEFT', 'B'], ['RIGHT', 'B']]
    _actions = []
    for action in actions:
        arr = np.array([False] * 12)
        for button in action:
            arr[buttons.index(button)] = True
        _actions.append(arr)
    
    return _actions