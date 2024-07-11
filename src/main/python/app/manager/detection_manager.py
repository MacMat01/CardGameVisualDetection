from pyzbar.pyzbar import decode


class DetectionManager:
    def __init__(self, card_detection_app):
        self.card_detection_app = card_detection_app

    def detect_and_process_qrcodes(self, frame):
        detected_qrcodes = self.detect_players(frame)
        self.card_detection_app.player_manager.process_qrcode(detected_qrcodes,
                                                              self.card_detection_app.round_manager.round_number,
                                                              self.card_detection_app.card_manager.cards_first_set,
                                                              self.card_detection_app.round_manager.first_phase_rounds)

    def detect_and_process_cards(self, frame):
        detect_result = self.card_detection_app.yolo_model_manager.detect_objects(frame)
        detected_cards_indices = detect_result[0].boxes.cls.tolist()
        detected_cards = [detect_result[0].names[i] for i in detected_cards_indices]
        self.card_detection_app.process_card_detection(detected_cards)
        self.card_detection_app.check_round_end()

    @staticmethod
    def detect_players(frame):
        return [obj.data.decode("utf-8").split(" has played")[0] for obj in decode(frame)]
