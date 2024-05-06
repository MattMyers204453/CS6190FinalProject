from datetime import datetime, timedelta
import numpy as np
import requests
import pickle
import os

API_TOKEN = ""


### LEAGUE ###
LEAGUE_ID = 271
### 2023-2024 SEASON ###
SEASON_2023_2024_ID = 21644
SEASON_2023_2024_REG_STAGE_ID = 77463996
SEASON_2023_2024_CHAMP_STAGE_ID = 77463995
SEASON_2023_2024_RELEG_STAGE_ID = 77463994
### 2022-2023 SEASON ###
SEASON_2022_2023_ID = 19686
SEASON_2022_2023_REG_STAGE_ID = 77457696
SEASON_2022_2023_CHAMP_STAGE_ID = 77457694
SEASON_2022_2023_RELEG_STAGE_ID = 77457695
PROMOTED_TEAM_VEJLE_ID = 7466
PROMOTED_TEAM_VEJLE_PLACING = 13
PROMOTED_TEAM_HVIDOVRE_ID = 8657
PROMOTED_TEAM_HVIDOVRE_PLACING = 14
### 2021-2022 SEASON ###
SEASON_2021_2022_ID = 18334
SEASON_2021_2022_REG_STAGE_ID = 77453568
SEASON_2021_2022_CHAMP_STAGE_ID = 77453566
SEASON_2021_2022_RELEG_STAGE_ID = 77453567
PROMOTED_TEAM_HORSENS_ID = 211
PROMOTED_TEAM_HORSENS_PLACING = 13
PROMOTED_TEAM_LYNGBY_ID = 2650
PROMOTED_TEAM_LYNGBY_PLACING = 14
### 2020-2021 SEASON ###
SEASON_2020_2021_ID = 17328
SEASON_2020_2021_REG_STAGE_ID = 77447994
SEASON_2020_2021_CHAMP_STAGE_ID = 77448541
SEASON_2020_2021_RELEG_STAGE_ID = 77448542
PROMOTED_TEAM_VIBORG_ID = 2447
PROMOTED_TEAM_VIBORG_PLACING = 13
PROMOTED_TEAM_SILKEBORG_ID = 86
PROMOTED_TEAM_SILKEBORG_PLACING = 14


### API ###
def get_player_info(player_name):
    url = f"https://api.sportmonks.com/v3/football/players/search/{player_name}?api_token={API_TOKEN}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['data']
    else:
        return f"Error: {response.status_code}"

def get_standings(season_id):
    url = f"https://api.sportmonks.com/v3/football/standings/seasons/{season_id}?api_token={API_TOKEN}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['data']
    else:
        return f"Error: {response.status_code}"
    

def get_team(team_id):
    url = f"https://api.sportmonks.com/v3/football/teams/{team_id}?api_token={API_TOKEN}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['data']
    else:
        return f"Error: {response.status_code}"
    
def get_all_teams_in_season(season_id):
    url = f"https://api.sportmonks.com/v3/football/seasons/{season_id}?api_token={API_TOKEN}&include=teams"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['data']['teams']
    else:
        return f"Error: {response.status_code}"
    
def get_all_teams_in_season_with_venues_for_each_team(season_id):
    url = f"https://api.sportmonks.com/v3/football/seasons/{season_id}?api_token={API_TOKEN}&include=teams;teams.venue"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['data']['teams']
    else:
        return f"Error: {response.status_code}"
    
def get_fixtures_for_season(season_id):
    url = f"https://api.sportmonks.com/v3/football/seasons/{season_id}?api_token={API_TOKEN}&include=fixtures;fixtures.scores"
    response = requests.get(url)
    if response.status_code == 200:
        fixtures = response.json()['data']['fixtures']
        fixtures_sorted_date = sorted(fixtures, key=lambda f: datetime.strptime(f['starting_at'], "%Y-%m-%d %H:%M:%S"))
        return fixtures_sorted_date
    else:
        return f"Error: {response.status_code}"
    
def get_matchup_history(team1_id, team2_id):
    url = f"https://api.sportmonks.com/v3/football/fixtures/head-to-head/{team1_id}/{team2_id}?api_token={API_TOKEN}&include="
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['data']
    else:
        return f"Error: {response.status_code}"
    
### PROCESSING ###
def build_teamID_to_teamName(season_id):
    teamID_to_teamName = {}
    teams = get_all_teams_in_season(season_id)
    for t in teams:
        teamID_to_teamName[t['id']] = t['name']
    teamName_to_teamID = {name: id for id, name in teamID_to_teamName.items()} # invert
    return teamID_to_teamName, teamName_to_teamID

def build_teamID_to_placing(season_id, champ_stage_id, releg_stage_id):
    teamID_to_placing = {}
    standings = get_standings(season_id)
    # standings_playoffs = [s for s in standings if s['stage_id'] == champ_stage_id or s['stage_id'] == releg_stage_id]
    # for s in standings_playoffs:
    for s in standings:
        team_id = s['participant_id']
        if s['stage_id'] == champ_stage_id:
            placing = s['position']
        elif s['stage_id'] == releg_stage_id:
            placing = s['position'] + 6
        else:
            continue
        teamID_to_placing[team_id] = placing
    # Get placings of two promoted teams from Division 1 (not Superliga)
    if season_id == SEASON_2022_2023_ID:
        teamID_to_placing[PROMOTED_TEAM_VEJLE_ID] = PROMOTED_TEAM_VEJLE_PLACING
        teamID_to_placing[PROMOTED_TEAM_HVIDOVRE_ID] = PROMOTED_TEAM_HVIDOVRE_PLACING
    elif season_id == SEASON_2021_2022_ID:
        teamID_to_placing[PROMOTED_TEAM_HORSENS_ID] = PROMOTED_TEAM_HORSENS_PLACING
        teamID_to_placing[PROMOTED_TEAM_LYNGBY_ID] = PROMOTED_TEAM_LYNGBY_PLACING
    elif season_id == SEASON_2020_2021_ID:
        teamID_to_placing[PROMOTED_TEAM_VIBORG_ID] = PROMOTED_TEAM_VIBORG_PLACING
        teamID_to_placing[PROMOTED_TEAM_SILKEBORG_ID] = PROMOTED_TEAM_SILKEBORG_PLACING
    teamID_to_placing_sorted = {id: p for id, p in sorted(teamID_to_placing.items(), key=lambda item: item[1])}
    return teamID_to_placing_sorted


def build_teamID_to_venueID(season_id):
    teamID_to_venueID = {}
    teams_with_their_venues = get_all_teams_in_season_with_venues_for_each_team(season_id)
    for t in teams_with_their_venues:
        teamID = t['id']
        venueID = t['venue']['id']
        teamID_to_venueID[teamID] = venueID
        # print(f"{t['name']} plays at {t['venue']['name']}")
    venueID_to_teamID = {name: id for id, name in teamID_to_venueID.items()}
    return teamID_to_venueID, venueID_to_teamID

def get_home_away_and_scores_from_fixture(fixture):
    scores = fixture['scores'] # score at first half, second half, and current (I believe current means final score for completed games)
    current_scores = [score for score in scores if score['description'] == 'CURRENT']
    home_team_id, away_team_id = None, None
    home_score, away_score = None, None
    for s in current_scores:
        if s['score']['participant'] == 'home':
            home_team_id = s['participant_id']
            home_score = s['score']['goals']
        elif s['score']['participant'] == 'away':
            away_team_id = s['participant_id']
            away_score = s['score']['goals']
    return home_team_id, home_score, away_team_id, away_score

# Return -1 if draw
def get_winning_team_id_from_fixture(fixture, teamName_to_teamID):
    result_info = fixture['result_info']
    if result_info is None:
        return None
    if 'draw' in result_info:
        return -1
    winning_team_name = result_info.split()[0]
    if winning_team_name == 'København' or winning_team_name == 'FC':
        winning_team_name = 'FC Copenhagen'
    winning_team_id = teamName_to_teamID[winning_team_name]
    return winning_team_id

def build_matchup_dict(season_id):
    _, teamName_to_teamID = build_teamID_to_teamName(season_id)
    upper_bound_date = None
    if season_id == SEASON_2023_2024_ID:
        upper_bound_date = datetime.strptime('2023-07-21', "%Y-%m-%d") # Start of 2022-2023 season
    if season_id == SEASON_2022_2023_ID:
        upper_bound_date = datetime.strptime('2022-07-15', "%Y-%m-%d") # Start of 2022-2023 season
    elif season_id == SEASON_2021_2022_ID:
        upper_bound_date = datetime.strptime('2021-07-16', "%Y-%m-%d") # Start of 2021-2022 season
    lower_bound_date = upper_bound_date - timedelta(days=730) # About 2 years earlier than start of current season
    historical_matchups_by_ids = {}
    teams = get_all_teams_in_season(season_id)
    for team1 in teams:
        id_1 = team1['id']
        for team2 in teams: 
            id_2 = team2['id']
            if id_1 == id_2:
                continue
            key = tuple(sorted((id_1, id_2)))
            if key not in historical_matchups_by_ids:
                historical_matchups_by_ids[key] = get_matchup_score(id_1, id_2, lower_bound_date, upper_bound_date, teamName_to_teamID)
    with open(f"{season_id}-matchups.pkl", "wb") as file:
        pickle.dump(historical_matchups_by_ids, file)
    

def get_matchup_score(team1_id, team2_id, lower_bound_date, upper_bound_date, teamName_to_teamID):
    low_id = team1_id if team1_id < team2_id else team2_id
    high_id = team1_id if team1_id > team2_id else team2_id
    matchups = get_matchup_history(team1_id, team2_id)
    matchups_last_2_years = [m for m in matchups if lower_bound_date <= datetime.strptime(m['starting_at'], "%Y-%m-%d %H:%M:%S") <= upper_bound_date 
                             and 'draw' not in m['result_info']]
    if len(matchups_last_2_years) == 0:
        return 0
    win_count_for_lower_id = 0
    for fixture in matchups_last_2_years:
        winning_id = get_winning_team_id_from_fixture(fixture, teamName_to_teamID)
        if winning_id == low_id:
            win_count_for_lower_id += 1
    return win_count_for_lower_id / len(matchups_last_2_years)



### UTIL ###
def print_placings(teamID_to_teamName, teamID_to_placing):
    for id, placing in teamID_to_placing.items():
        print(f"{teamID_to_teamName[id]} got {placing} place")





### DATASET BUILDING ###
def construct_X_and_Y(season_id, regular_stage_id):
    teamID_to_teamName, teamName_to_teamID = build_teamID_to_teamName(season_id)
    teamID_to_placings = None
    if season_id == SEASON_2023_2024_ID:
        teamID_to_placings = build_teamID_to_placing(SEASON_2022_2023_ID, SEASON_2022_2023_CHAMP_STAGE_ID, SEASON_2022_2023_RELEG_STAGE_ID)
    elif season_id == SEASON_2022_2023_ID:
        teamID_to_placings = build_teamID_to_placing(SEASON_2021_2022_ID, SEASON_2021_2022_CHAMP_STAGE_ID, SEASON_2021_2022_RELEG_STAGE_ID)
    elif season_id == SEASON_2021_2022_ID:
        teamID_to_placings = build_teamID_to_placing(SEASON_2020_2021_ID, SEASON_2020_2021_CHAMP_STAGE_ID, SEASON_2020_2021_RELEG_STAGE_ID)

    matchup_scores = {}
    if os.path.isfile(f"{season_id}-matchups.pkl"):
        print('SAVED MATCHUPS FOUND')
        with open(f"{season_id}-matchups.pkl", "rb") as file:
            matchup_scores = pickle.load(file)
    else:
        print('BUILDING MATCHUPS')
        build_matchup_dict(season_id)
        with open(f"{season_id}-matchups.pkl", "rb") as file:
            matchup_scores = pickle.load(file)
    
    result_history_by_team_id = {}
    fixtures = get_fixtures_for_season(season_id)
    feat_vect_list = []
    label_list = []
    for f in fixtures:
        feat_vec, label = convert_fixture_to_feature_vector(f, teamName_to_teamID, teamID_to_placings, result_history_by_team_id, matchup_scores)
        if feat_vec is None:
            continue
        feat_vect_list.append(feat_vec)
        label_list.append(label)
    X = np.array(feat_vect_list)
    Y = np.array(label_list)
    return X, Y

def convert_fixture_to_feature_vector(fixture, teamName_to_teamID, teamID_to_placings, result_history_by_team_id, matchup_scores):
    home_id, home_score, away_id, away_score = get_home_away_and_scores_from_fixture(fixture)
    winning_id = get_winning_team_id_from_fixture(fixture, teamName_to_teamID)
    if winning_id is None:
        return None, None

    losing_id = -1
    if winning_id != -1:
        losing_id = away_id if home_id == winning_id else home_id

    matchup_key = tuple(sorted((home_id, away_id)))
    matchup_score = matchup_scores[matchup_key]
    update_recent_performance(result_history_by_team_id, home_id, away_id, winning_id)

    # Features
    RECENT_PERF_DEPTH = 5
    home_team_matchup_score = matchup_score if home_id == matchup_key[0] else (1 - matchup_score)
    home_team_placing = normalize_placing(teamID_to_placings[home_id])
    home_team_recent_perf_score = recent_performance_score(home_id, result_history_by_team_id, RECENT_PERF_DEPTH)
    away_team_placing = normalize_placing(teamID_to_placings[away_id])
    away_team_recent_perf_score = recent_performance_score(away_id, result_history_by_team_id, RECENT_PERF_DEPTH)

    feature_vector = np.array([home_team_matchup_score, home_team_placing, home_team_recent_perf_score,
                                away_team_placing, away_team_recent_perf_score])
    label = 1 if winning_id == home_id else 0

    return feature_vector, label

def update_recent_performance(result_history_by_team_id, home_id, away_id, winning_id):
    if winning_id == -1:
        if home_id in result_history_by_team_id:
            result_history_by_team_id[home_id].append('n')
        else:
            result_history_by_team_id[home_id] = ['n']

        if away_id in result_history_by_team_id:
            result_history_by_team_id[away_id].append('n')
        else:
            result_history_by_team_id[away_id] = ['n']
    else: 
        losing_id = away_id if winning_id == home_id else home_id
        if winning_id in result_history_by_team_id:
            result_history_by_team_id[winning_id].append('w')
        else:
            result_history_by_team_id[winning_id] = ['w']

        if losing_id in result_history_by_team_id:
            result_history_by_team_id[losing_id].append('n')
        else:
            result_history_by_team_id[losing_id] = ['n']

def recent_performance_score(team_id, result_history_by_team_id, depth):
    result_history = result_history_by_team_id[team_id]
    if len(result_history) < 2:
        return 0.0
    recent_games = result_history[-(depth + 1):-1] # don't include most recent game, otherwise we are encoding the label into the feature vector
    return recent_games.count('w') / len(recent_games)

def recent_performance_score_decay(team_id, result_history_by_team_id, depth):
    result_history = result_history_by_team_id[team_id]
    if len(result_history) < 2:
        return 0.0
    recent_games = result_history[-(depth + 1):-1] # don't include most recent game, otherwise we are encoding the label into the feature vector
    weights = [0.1, 0.15, 0.2, 0.25, 0.3]  # more recent games have higher weights
    score = 0.0
    used_weights = weights[-len(recent_games):]
    
    for i, outcome in enumerate(recent_games):
        if outcome == 'w':
            score += used_weights[i]
        else:
            score -= used_weights[i]
    
    # min-max normailization
    max_score = sum(used_weights)
    min_score = -max_score
    return (score - min_score) / (max_score - min_score)


# Normalize placing to [0, 1]. 1 -> 1 and 14-> 0
def normalize_placing(placing):
    LOWEST_PLACING = 14 # 1-10 from last Superliga season and 13-14 from Division 1 promotions. Teams placing 11 and 12 are relegated and not in this season
    return (LOWEST_PLACING - placing) / 13


X_TRAIN_1, Y_TRAIN_1 = construct_X_and_Y(SEASON_2021_2022_ID, SEASON_2021_2022_REG_STAGE_ID)
np.save("X-TRAIN-1.npy", X_TRAIN_1)
np.save("Y-TRAIN-1.npy", Y_TRAIN_1)

X_TRAIN_2, Y_TRAIN_2 = construct_X_and_Y(SEASON_2022_2023_ID, SEASON_2022_2023_REG_STAGE_ID)
np.save("X-TRAIN-2.npy", X_TRAIN_2)
np.save("Y-TRAIN-2.npy", Y_TRAIN_2)

X_TEST, Y_TEST = construct_X_and_Y(SEASON_2023_2024_ID, SEASON_2023_2024_REG_STAGE_ID)
np.save("X-TEST.npy", X_TEST)
np.save("Y-TEST.npy", Y_TEST)

# build_matchup_dict(SEASON_2022_2023_ID)







