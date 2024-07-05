import csv
import os
from datetime import datetime

from influxdb_client import InfluxDBClient, Point


class InfluxDBManager:
    """
    A class used to manage interactions with InfluxDB.

    Attributes
    ----------
    client : InfluxDBClient
        a client object to interact with InfluxDB
    write_api : WriteApi
        an API object to write data to InfluxDB

    Methods
    -------
    write_to_influxdb(player, card, elapsed_time):
         Write the player, card, and elapsed time data to InfluxDB.
    """

    def __init__(self):
        """
        Constructs a new InfluxDBManager.

        The InfluxDBManager will use the InfluxDB client with the specified URL, token, and organization.
        """
        self.client = InfluxDBClient(url="https://us-east-1-1.aws.cloud2.influxdata.com",
                                     token=os.getenv("INFLUXDB_TOKEN"), org="Cris-and-Matt")
        self.write_api = self.client.write_api()

    def write_to_influxdb(self, player, card, elapsed_time, round_number, current_matchups):
        """
        Writes the player, card, and elapsed time data to InfluxDB.

        The data is written as a point with the player and card as tags and the elapsed time as a field.

        Parameters
        ----------
        player : str
            The player who played the card.
        card : str
            The card that was played.
        elapsed_time : float
            The time elapsed since the player's last move.
        round_number : int
            The current round number.
        current_matchups : list
            The current matchups between players.
        """
        phase = None
        vs = None

        if 1 <= round_number <= 1:  # 1 <= round_number <= 12
            phase = 1
        elif 1 < round_number <= 2:  # 12 < round_number <= 18
            phase = 2
        elif 2 < round_number <= 3:  # 18 < round_number <= 24
            phase = 3

        for matchup in current_matchups:
            if player in matchup:
                vs = matchup[0] if matchup[1] == player else matchup[1]
                break

        # Remove non-digit characters from the card string
        card = ''.join(c for c in card if c.isdigit())

        point = Point("Card Game").tag("Phase", phase).tag("Round", round_number).tag("Player", player).tag("Card",
                                                                                                            card).tag(
            "VS", vs).field("Thinking Time", elapsed_time)
        print(f"Writing to InfluxDB: {point.to_line_protocol()}")
        self.write_api.write(bucket="StrategicFruitsData", org="Cris-and-Matt", record=point.to_line_protocol())
