import time

import cv2

from src.main.python.app.manager.card_manager import CardManager
from src.main.python.app.manager.detection_manager import DetectionManager
from src.main.python.app.manager.player_manager import PlayerManager
from src.main.python.app.manager.round_manager import RoundManager
from src.main.python.app.manager.video_capture_manager import VideoCaptureManager
from src.main.python.app.manager.yolo_model_manager import YOLOModelManager


class CardDetectionApp:
    def __init__(self, video_file=None):
        self.video_capture_manager = VideoCaptureManager(video_file)
        self.yolo_model_manager = YOLOModelManager()
        self.player_manager = PlayerManager(self)
        self.card_manager = CardManager()
        self.round_manager = RoundManager(self)
        self.detection_manager = DetectionManager(self)
        self.player_detected = False
        self.cards_detected = True
        self.round_manager.setup_round_robin()
        print(f"Round {self.round_manager.round_number} starting.")
        print(f"Matchups: {self.round_manager.current_matchups}")
        self.start_time = time.time()

    def get_elapsed_time(self):
        elapsed_time = time.time() - self.start_time
        print(f"Elapsed time: {elapsed_time}")
        return elapsed_time

    def check_round_end(self):
        if (self.round_manager.round_number <= self.round_manager.first_phase_rounds and len(
                self.card_manager.cards_first_set) == 4 and self.player_detected and not self.cards_detected):
            self.round_manager.end_round()
        elif (self.round_manager.round_number > self.round_manager.first_phase_rounds and len(
                self.card_manager.cards_second_set) == 4 and self.player_detected and not self.cards_detected):
            self.round_manager.end_round()
        if (
                self.round_manager.round_number > self.round_manager.first_phase_rounds and self.player_detected and not self.cards_detected and len(
            self.card_manager.cards_second_set) < 4):
            self.card_manager.duplicate_cards()

    def process_card_detection(self, detected_cards):
        self.card_manager.detect_card_played(detected_cards, self.player_manager.players_first_set)
        self.check_round_end()

    def match_and_record_players_cards(self):
        matched_players_cards = []

        for players_set in [self.player_manager.players_first_set, self.player_manager.players_second_set]:
            for player, player_time in players_set:
                match = next(
                    (card for card_set in [self.card_manager.cards_first_set, self.card_manager.cards_second_set] for
                     card in card_set if self.is_match(player, card, matched_players_cards)), None)
                if match:
                    matched_players_cards.append((player, player_time, match))

        return matched_players_cards

    @staticmethod
    def is_match(player, card, matched_players_cards):
        return player[0].lower() == card[-1].lower() and (player, card) not in matched_players_cards

    def process_frame(self):
        ret, frame = self.video_capture_manager.read_frame()
        if not ret:
            return False
        self.detection_manager.detect_and_process_qrcodes(frame)
        detect_result = self.yolo_model_manager.detect_objects(frame)
        detect_image = detect_result[0].plot()
        detect_players = DetectionManager.detect_players(frame)
        if detect_players:
            self.player_detected = True
        detected_cards_indices = detect_result[0].boxes.cls.tolist()
        detected_cards = [detect_result[0].names[i] for i in detected_cards_indices]
        self.cards_detected = bool(detected_cards)
        self.detection_manager.detect_and_process_cards(frame)
        cv2.imshow('Card Detection', detect_image)
        return True

    def run(self):
        while True:
            if not self.process_frame() or cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.video_capture_manager.release()
        cv2.destroyAllWindows()
        self.round_manager.write_round_data_to_csv()


if __name__ == "__main__":
    app = CardDetectionApp()
    app.run()
