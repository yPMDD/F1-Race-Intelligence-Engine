# 2026 Season Grid Configuration
# This overrides historical database records to reflect the actual 2026 lineup.

GRID_2026 = [
    {"driver_id": "VER", "name": "Max Verstappen", "team": "Red Bull Racing"},
    {"driver_id": "HAD", "name": "Isack Hadjar", "team": "Red Bull Racing"},
    {"driver_id": "HAM", "name": "Lewis Hamilton", "team": "Ferrari"},
    {"driver_id": "LEC", "name": "Charles Leclerc", "team": "Ferrari"},
    {"driver_id": "NOR", "name": "Lando Norris", "team": "McLaren"},
    {"driver_id": "PIA", "name": "Oscar Piastri", "team": "McLaren"},
    {"driver_id": "RUS", "name": "George Russell", "team": "Mercedes"},
    {"driver_id": "ANT", "name": "Kimi Antonelli", "team": "Mercedes"},
    {"driver_id": "ALO", "name": "Fernando Alonso", "team": "Aston Martin"},
    {"driver_id": "STR", "name": "Lance Stroll", "team": "Aston Martin"},
    {"driver_id": "SAI", "name": "Carlos Sainz", "team": "Williams"},
    {"driver_id": "ALB", "name": "Alex Albon", "team": "Williams"},
    {"driver_id": "GAS", "name": "Pierre Gasly", "team": "Alpine"},
    {"driver_id": "DOO", "name": "Jack Doohan", "team": "Alpine"},
    {"driver_id": "BOR", "name": "Gabriel Bortoleto", "team": "Audi (Alfa Romeo)"},
    {"driver_id": "HUL", "name": "Nico Hulkenberg", "team": "Audi (Alfa Romeo)"},
    {"driver_id": "TSU", "name": "Yuki Tsunoda", "team": "RB"},
    {"driver_id": "LAW", "name": "Liam Lawson", "team": "RB"},
    {"driver_id": "BEA", "name": "Oliver Bearman", "team": "Haas"},
    {"driver_id": "OCO", "name": "Esteban Ocon", "team": "Haas"},
    {"driver_id": "BOT", "name": "Valtteri Bottas", "team": "Cadillac Racing"},
    {"driver_id": "PER", "name": "Sergio Perez", "team": "Cadillac Racing"},
]

# Helper to get team by driver
def get_2026_team(driver_id):
    for entry in GRID_2026:
        if entry["driver_id"] == driver_id:
            return entry["team"]
    return "Unknown"
