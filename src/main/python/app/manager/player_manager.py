class PlayerManager:

    def __init__(self, app):
        self.players_first_set = []
        self.players_second_set = []
        self.app = app

    def process_qrcode(self, detected_qrcodes, round_number, cards, first_phase_rounds):
        for qrcode in detected_qrcodes:
            if not any(player == qrcode for player, _ in self.players_first_set):
                self.players_first_set.append((qrcode, round(self.app.get_elapsed_time(), 2)))
                print(f"{qrcode} has played.")
            if not any(player == qrcode for player, _ in
                       self.players_second_set) and round_number > first_phase_rounds and len(cards) == 4:
                self.players_second_set.append((qrcode, round(self.app.get_elapsed_time(), 2)))
                print(f"{qrcode} has played.")
