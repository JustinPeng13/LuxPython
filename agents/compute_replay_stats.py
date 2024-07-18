import os
import json

def get_json_file_paths(directory):
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

directory = './replays/'
json_files = get_json_file_paths(directory)

results = {}

for filename in json_files:
    with open(filename, 'r') as file:
        data = json.load(file)

    width, height = data['width'], data['height']
    if (width, height) not in results:
        results[(width, height)] = {}
    stats = results[(width, height)]

    teams = data['teamDetails'][0]['name'], data['teamDetails'][1]['name']
    team0, team1 = teams[0], teams[1]
    if team0 not in stats:
        stats[team0] = {'win': 0, 'lose': 0, 'research': [], 'units': [], 'cityCells': [], 'max_units': [], 'max_city_tiles': []}
    if team1 not in stats:
        stats[team1] = {'win': 0, 'lose': 0, 'research': [], 'units': [], 'cityCells': [], 'max_units': [], 'max_city_tiles': []}

    last_turn = data['stateful'][-1]

    team0_research = last_turn['teamStates']['0']['researchPoints']
    team1_research = last_turn['teamStates']['1']['researchPoints']
    stats[team0]['research'].append(team0_research)
    stats[team1]['research'].append(team1_research)

    team0_max_units = 0
    team1_max_units = 0
    team0_max_city_tiles = 0
    team1_max_city_tiles = 0
    for turn in data['stateful']:
        team0_max_units = max(team0_max_units, len(turn['teamStates']['0']['units']))
        team1_max_units = max(team1_max_units, len(turn['teamStates']['1']['units']))
        cities = turn['cities']
        team0_cities = [cities[key] for key in cities if cities[key]['team'] == 0]
        team1_cities = [cities[key] for key in cities if cities[key]['team'] == 1]
        team0_max_city_tiles = max(team0_max_city_tiles, sum(len(team0_cities[i]['cityCells']) for i in range(len(team0_cities))))
        team1_max_city_tiles = max(team1_max_city_tiles, sum(len(team1_cities[i]['cityCells']) for i in range(len(team1_cities))))
    stats[team0]['max_units'].append(team0_max_units)
    stats[team1]['max_units'].append(team1_max_units)
    stats[team0]['max_city_tiles'].append(team0_max_city_tiles)
    stats[team1]['max_city_tiles'].append(team1_max_city_tiles)

    cities = last_turn['cities']
    team0_cities = [cities[key] for key in cities if cities[key]['team'] == 0]
    team1_cities = [cities[key] for key in cities if cities[key]['team'] == 1]
    team0_num_city_tiles = sum(len(team0_cities[i]['cityCells']) for i in range(len(team0_cities)))
    team1_num_city_tiles = sum(len(team1_cities[i]['cityCells']) for i in range(len(team1_cities)))
    stats[team0]['cityCells'].append(team0_num_city_tiles)
    stats[team1]['cityCells'].append(team1_num_city_tiles)

    team0_num_units = len(last_turn['teamStates']['0']['units'])
    team1_num_units = len(last_turn['teamStates']['1']['units'])

    winner = None
    if team0_num_city_tiles == team1_num_city_tiles:
        if team0_num_units == team1_num_units:
            winner = None # draw
        else:
            winner = int(team1_num_units > team0_num_units)
    else:
        winner = int(team1_num_city_tiles > team0_num_city_tiles)
    if winner is not None:
        stats[teams[winner]]['win'] += 1
        stats[teams[not winner]]['lose'] += 1
    else:
        print("Draw found:", file)
        assert 0

def to2dp(x):
    return round(x, 2)

for map_size,stats in sorted(results.items()):
    print("Map:", map_size)
    for k,v in sorted(stats.items()):
        print(k)
        games = v['win'] + v['lose']
        wr = v['win'] / games * 100
        print("games:", games)
        print("winrate:", to2dp(wr))
        print("mean ending city tiles:", to2dp(sum(v['cityCells']) / len(v['cityCells'])))
        print('mean max city tiles:', to2dp(sum(v['max_city_tiles']) / len(v['max_city_tiles'])))
        print("mean max units:", to2dp(sum(v['max_units']) / len(v['max_units'])))
        print("mean research pts:", to2dp(sum(v['research']) / len(v['research'])))
        print()
