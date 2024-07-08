import csv
import time
from datetime import datetime


class RoundManager:
    def __init__(self, card_detection_app):
        self.card_detection_app = card_detection_app
        self.first_phase_rounds = 1
        self.round_number = 12
        self.current_phase = 1
        self.current_matchups = None
        self.round_data = []

    def increment_round(self):
        self.round_number += 1
        self.increment_phase()
        self.setup_round_robin()

    def increment_phase(self):
        if 1 <= self.round_number <= 12:  # 1 <= self.round_number <= 12
            self.current_phase = 1
        elif 12 < self.round_number <= 18:  # 12 < self.round_number <= 18
            self.current_phase = 2
        elif 18 < self.round_number <= 24:  # 18 < self.round_number <= 24
            self.current_phase = 3
        print(f"Phase {self.current_phase} and Round {self.round_number} starting.")

    def setup_round_robin(self):
        matchups = {1: [("Apple", "Pear"), ("Orange", "Banana")], 2: [("Apple", "Banana"), ("Orange", "Pear")],
                    3: [("Pear", "Banana"), ("Orange", "Apple")]}
        round_number = self.round_number % 3
        if round_number == 0:
            round_number = 3
        self.current_matchups = matchups[round_number]

    def write_round_data_to_csv(self):
        current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        csv_file_name = f'strategic_fruits_card_game_playtest_{current_time}.csv'
        header = ['Phase', 'Round', 'Player', 'Card', 'VS', 'Thinking Time']
        try:
            with open(csv_file_name, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                for data in self.round_data:
                    writer.writerow(
                        [data['Phase'], data['Round'], data['Player'], data['Card'], data['VS'], data['Thinking Time']])
        except Exception as e:
            print(f"Error writing to CSV: {e}")

    def end_round(self):
        print(f"Round {self.round_number} ended")
        matched_players_cards = self.card_detection_app.match_and_record_players_cards()
        for player, player_time, card in matched_players_cards:
            if (player, player_time) in self.card_detection_app.player_manager.players_first_set:
                self.card_detection_app.player_manager.players_first_set.remove((player, player_time))
            elif (player, player_time) in self.card_detection_app.player_manager.players_second_set:
                self.card_detection_app.player_manager.players_second_set.remove((player, player_time))
            if card in self.card_detection_app.card_manager.cards_first_set:
                self.card_detection_app.card_manager.cards_first_set.remove(card)
            elif card in self.card_detection_app.card_manager.cards_second_set:
                self.card_detection_app.card_manager.cards_second_set.remove(card)
            self.card_detection_app.match_and_record_players_cards()
            vs = self.find_vs_player(player)
            card = ''.join(c for c in card if c.isdigit())
            self.round_data.append(
                {'Phase': self.current_phase, 'Round': self.round_number, 'Player': player, 'Card': card, 'VS': vs,
                 'Thinking Time': player_time})
            print(self.round_data[-1]) 
        self.clear_sets()
        self.increment_round()
        self.card_detection_app.start_time = time.time()

    def clear_sets(self):
        self.card_detection_app.player_manager.players_first_set.clear()
        self.card_detection_app.player_manager.players_second_set.clear()
        self.card_detection_app.card_manager.cards_first_set.clear()
        self.card_detection_app.card_manager.cards_second_set.clear()
        self.card_detection_app.card_manager.detected_cards_counts.clear()

    def find_vs_player(self, player):
        for matchup in self.current_matchups:
            if player in matchup:
                return matchup[0] if matchup[1] == player else matchup[1]
        return None
