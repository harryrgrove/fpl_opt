from definitions import root_dir
import numpy as np
from typing import List
import json
import requests
from unidecode import unidecode
import pandas as pd


class Projections:
    def __init__(self, projections):
        self.projections = projections

    @classmethod
    def get_fplreview(cls, season, next_gw):
        df = pd.read_csv(root_dir + f'/data/fplreview/{season}/GW{next_gw}.csv')
        df.index = df.index.rename('ID')
        return Projections(df)

    @property
    def loc(self):
        return self.projections.loc

    @property
    def index(self):
        return self.projections.index

    @property
    def columns(self):
        return self.projections.columns

    def __getitem__(self, item):
        return self.projections[item]

    def __setitem__(self, key, value):
        self.projections[key] = value

    def __str__(self):
        return str(self.projections)

    def perturb(self, method=None, params=None):
        pass

    def get_player(self, name, team=None, position=None):
        """
        :param name: Name of player (generally surname)
        :param team: Optional (recommended if common name), team of player
        :param position:Optional (recommended if common name), position of player from ('G', 'D', 'M', 'F')
        :return: ID of player
        """
        proj_filter = pd.Series(np.repeat(1, len(self.projections)))
        proj_filter.index = self.projections.index
        if team is not None:
            proj_filter *= self.projections['Team'] == team
        if position is not None:
            proj_filter *= self.projections['Pos'] == position

        valid_players = self.projections[self.projections['Name'].apply(unidecode) == name]
        if len(valid_players) == 0:
            valid_players = self.projections[self.projections['Name'] == name]
        if len(valid_players) != 1:
            raise Exception('Invalid player selection')
        return valid_players.index[0]


class Squad:
    def __init__(self, players: List[int], itb: float = 0, fts: int = 1, active_chip: str = None):
        self.players, self.itb, self.fts, self.active_chip = players, itb, fts, active_chip

    @classmethod
    def from_id(cls, fpl_id, gw=None):
        # Get picks info by FPL ID
        session = requests.Session()
        api_path = 'https://fantasy.premierleague.com/api'
        current_gw = json.loads(session.get(f'{api_path}/entry/{fpl_id}/').text)['current_event']
        if gw is None:
            gw = current_gw
        response = session.get(f'{api_path}/entry/{fpl_id}/event/{gw}/picks/#/')
        data = json.loads(response.text)

        fts = 1
        if gw != 1:
            response = session.get(f'{api_path}/entry/{fpl_id}/history/#/')
            history = json.loads(response.text)
            wildcards = [chip['event'] for chip in history['chips'] if chip['name'] == 'wildcard']
            freehit = [chip['event'] for chip in history['chips'] if chip['name'] == 'freehit']
            fts_available = []
            if len(wildcards):
                ranges = [range(1, min(wildcards)), range(min(wildcards), max(wildcards)),
                          range(max(wildcards), gw + 1)]
            else:
                ranges = [range(1, gw + 1)]
            for r in ranges:
                ft_pre_deadline = 0
                for past_gw in r:
                    if past_gw in freehit:
                        ft_pre_deadline -= 1
                    transfer_count = history['current'][past_gw - 1]['event_transfers']
                    ft_post_deadline = min(max(ft_pre_deadline + 1 - transfer_count, 1), 2)
                    fts_available.append(ft_post_deadline)
                    ft_pre_deadline = ft_post_deadline
            fts = fts_available[gw - 1]
        print(data['picks'])
        return Squad(
            players=[d['element'] - 1 for d in data['picks']],
            itb=data['entry_history']['bank'] / 10,
            active_chip=data['active_chip'],
            fts=fts
        )

    def __str__(self):
        output = f'Players: {self.players}'
        output += f'\nIn the bank: {self.itb}'
        output += f'\nFree Transfers Available: {self.fts}'
        return output


if __name__ == '__main__':
    print(Squad.from_id(1433))
