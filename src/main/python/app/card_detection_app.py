import csv
import time
from datetime import datetime

import cv2
from pyzbar.pyzbar import decode

from manager.card_manager import CardManager
from manager.influxdb_manager import InfluxDBManager
from manager.player_manager import PlayerManager
from manager.video_capture_manager import VideoCaptureManager
from manager.yolo_model_manager import YOLOModelManager


class CardDetectionApp:
    def __init__(self, video_file=None):
        self.current_matchups = None
        self.influxdb_manager = InfluxDBManager()
        self.video_capture_manager = VideoCaptureManager(video_file)
        self.yolo_model_manager = YOLOModelManager()
        self.player_manager = PlayerManager(self)
        self.card_manager = CardManager()
        self.first_phase_rounds = 1
        self.round_number = 1
        self.current_phase = 1
        self.player_detected = False
        self.cards_detected = True
        self.setup_round_robin()
        print(f"Round {self.round_number} starting.")
        print(f"Matchups: {self.current_matchups}")
        self.start_time = time.time()
        self.round_data = []

    def get_elapsed_time(self):
        elapsed_time = time.time() - self.start_time
        print(f"Elapsed time: {elapsed_time}")
        return elapsed_time

    def increment_round(self):
        self.round_number += 1
        self.increment_phase()
        self.setup_round_robin()

    def increment_phase(self):
        if 1 <= self.round_number <= 1:
            self.current_phase = 1
        elif 1 < self.round_number <= 2:
            self.current_phase = 2
        elif 2 < self.round_number <= 3:
            self.current_phase = 3
        print(f"Phase {self.current_phase}.")
        print(f"Round {self.round_number} starting.")

    def check_round_end(self):
        if (self.round_number <= self.first_phase_rounds and len(
                self.card_manager.cards_first_set) == 4 and self.player_detected and not self.cards_detected):
            self.end_round()
        elif (self.round_number > self.first_phase_rounds and len(
                self.card_manager.cards_second_set) == 4 and self.player_detected and not self.cards_detected):
            self.end_round()
        if (self.round_number > self.first_phase_rounds and self.player_detected and not self.cards_detected and len(
                self.card_manager.cards_second_set) < 4):
            self.card_manager.duplicate_cards()

    def end_round(self):
        vs = None
        print(f"Round {self.round_number} ended")
        matched_players_cards = self.write_to_influxdb()
        for player, player_time, card in matched_players_cards:
            if (player, player_time) in self.player_manager.players_first_set:
                self.player_manager.players_first_set.remove((player, player_time))
            elif (player, player_time) in self.player_manager.players_second_set:
                self.player_manager.players_second_set.remove((player, player_time))
            if card in self.card_manager.cards_first_set:
                self.card_manager.cards_first_set.remove(card)
            elif card in self.card_manager.cards_second_set:
                self.card_manager.cards_second_set.remove(card)
            self.influxdb_manager.write_to_influxdb(player, card, player_time, self.round_number, self.current_matchups)
            # Remove non-digit characters from the card string
            card = ''.join(c for c in card if c.isdigit())

            for matchup in self.current_matchups:
                if player in matchup:
                    vs = matchup[0] if matchup[1] == player else matchup[1]
                    break
            self.round_data.append(
                {'Phase': self.current_phase, 'Round': self.round_number, 'Player': player, 'Card': card, 'VS': vs,
                 'Thinking Time': player_time})
        self.player_manager.players_first_set.clear()
        self.player_manager.players_second_set.clear()
        self.card_manager.cards_first_set.clear()
        self.card_manager.cards_second_set.clear()
        self.card_manager.detected_cards_counts.clear()
        self.increment_round()
        self.start_time = time.time()

    def write_round_data_to_csv(self):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_file_name = f'round_data_{current_time}.csv'
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

    def process_card_detection(self, detected_cards):
        self.card_manager.detect_card_played(detected_cards, self.player_manager.players_first_set)
        self.check_round_end()

    def detect_and_process_qrcodes(self, frame):
        detected_qrcodes = self.detect_players(frame)
        self.player_manager.process_qrcode(detected_qrcodes, self.round_number, self.card_manager.cards_first_set,
                                           self.first_phase_rounds)

    def detect_and_process_cards(self, frame):
        detect_result = self.yolo_model_manager.detect_objects(frame)
        detected_cards_indices = detect_result[0].boxes.cls.tolist()
        detected_cards = [detect_result[0].names[i] for i in detected_cards_indices]
        self.process_card_detection(detected_cards)
        self.check_round_end()

    def write_to_influxdb(self):
        matched_players_cards = []

        def match_players_cards(players, cards):
            for player, player_time in players:
                for card in cards:
                    if player[0].lower() == card[-1].lower() and (
                            player, player_time, card) not in matched_players_cards:
                        matched_players_cards.append((player, player_time, card))
                        break

        match_players_cards(self.player_manager.players_first_set, self.card_manager.cards_first_set)
        match_players_cards(self.player_manager.players_second_set, self.card_manager.cards_second_set)
        return matched_players_cards

    def process_frame(self):
        ret, frame = self.video_capture_manager.read_frame()
        if not ret:
            return False
        self.detect_and_process_qrcodes(frame)
        detect_result = self.yolo_model_manager.detect_objects(frame)
        detect_image = detect_result[0].plot()
        detect_players = self.detect_players(frame)
        if detect_players:
            self.player_detected = True
        detected_cards_indices = detect_result[0].boxes.cls.tolist()
        detected_cards = [detect_result[0].names[i] for i in detected_cards_indices]
        self.cards_detected = bool(detected_cards)
        self.detect_and_process_cards(frame)
        cv2.imshow('Card Detection', detect_image)
        return True

    def run(self):
        while True:
            if not self.process_frame() or cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.video_capture_manager.release()
        cv2.destroyAllWindows()
        self.write_round_data_to_csv()

    @staticmethod
    def detect_players(frame):
        return [obj.data.decode("utf-8").split(" has played")[0] for obj in decode(frame)]

    def setup_round_robin(self):
        matchups = {1: [("Apple", "Pear"), ("Orange", "Banana")], 2: [("Apple", "Banana"), ("Orange", "Pear")],
                    3: [("Pear", "Banana"), ("Orange", "Apple")]}
        round_number = self.round_number % 3
        if round_number == 0:
            round_number = 3
        self.current_matchups = matchups[round_number]


if __name__ == "__main__":
    app = CardDetectionApp()
    app.run()
