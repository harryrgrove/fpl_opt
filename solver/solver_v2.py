from lp_tools import LpProblem, DecisionSeries, DecisionMatrix
from definitions import root_dir
from typing import List, Callable, Dict
import pandas as pd
import numpy as np
import pulp


def get_fplreview(season, next_gw):
    """
    :param season: Starting year of relevant season
    :param next_gw: Earliest gameweek contained in projections
    :return: pd.Dataframe object with columns '{gw}_pts' for each gw in projections
    """
    df = pd.read_csv(root_dir + f'/data/fplreview/{season}/GW{next_gw}.csv')
    df.index = df.index.rename('ID')
    return df


class Squad:
    def __init__(self, players: List[int], itb: float = 0, fts: int = 1, active_chip: str = None):
        self.players, self.itb, self.fts, self.active_chip = players, itb, fts, active_chip

    @classmethod
    def from_id(cls, fpl_id, gw):
        pass


class Solver:
    """
    Framework for Fantasy Premier League linear optimisation program
    ================================================================
    Credit to @sertalpbilal for inspiration
    """
    def __init__(
            self,
            horizon: int = 8,
            bench_weights: List[List[float]] = None,
            vc_weight: float = 0,
            penalties: Dict[Callable, float] = None,
            budget_decay_rate: int = 0,
            transfer_pattern: List[int] = None,
            exclude_everton: bool = False
    ):
        """
        :param horizon: Number of future GWs included in optimisation, max=8
        :param bench_weights: [bench_gk_weight, 1st_bench_weight, 2nd_bench_weight, 3rd_bench_weight]
        :param vc_weight: Extra weight of vice captain in objective function
        :param penalties: Dict of penalty functions taking self, current objective and parameter
        :param budget_decay_rate: Linear decay parameter; budget reduces by budget_decay_rate each future GW
        :param transfer_pattern: Manual override for transfer logic, nth element is transfers in n GWs from now
        :param exclude_everton: If True, no Everton players are included in solutions
        """
        self.horizon = horizon
        if bench_weights is None:
            self.bench_weights = np.array([
                [0.001, 0.15,   0.01,   0.001],
                [0.01,  0.22,   0.03,   0.004],
                [0.02,  0.26,   0.045,  0.006],
                [0.03,  0.36,   0.075,  0.016],
                [0.035, 0.40,   0.095,  0.02],
                [0.04,  0.43,   0.12,   0.03],
                [0.043, 0.46,   0.14,   0.035],
                [0.045, 0.48,   0.15,   0.04]
            ][:horizon])
        else:
            self.bench_weights = bench_weights
        self.vc_weight = vc_weight
        if penalties is None:
            self.penalties = []
        else:
            self.penalties = penalties
        self.budget_decay_rate = budget_decay_rate
        self.transfer_pattern = transfer_pattern
        self.exclude_everton = exclude_everton

        self.rolls = None
        self.force_chips = None

    def solve(
            self,
            projections: pd.DataFrame,
            initial_squad: Squad,
            next_gw: int = 1,
            force_chips: Dict[int, str] = None,
            force_players: Dict[str, list] = None,
            force_transfers: Dict[int, dict] = None
    ):
        """
        Method for solving specific instance of FPL problem
        :param projections: DataFrame containing projections, such as from get_fplreview output
        :param next_gw: Next GW
        :param initial_squad: Squad object with data for current itb, fts and active chip
        :param force_chips: {gw_a: wildcard, gw_b: bench_boost, gw_c: triple_captain, gw_d: free_hit} forces wildcard
            in gw_a, ...m free hit in gw_d.
        :param force_players: {'include': [{players to include}], 'exclude': [{players to exclude}]}
        :param force_transfers: {gw_a: {'in': [{player list 1}], 'out': [{player list 2}]}, gw_b: ...} forces players in
            player list 1 to be transferred in for players in player list 2 in gw_a
        :return: class Solution containing transfer and team selection data for optimal solution
        """
        # Set up useful references
        initial_players = initial_squad.players
        initial_itb = initial_squad.itb
        initial_fts = initial_squad.fts
        active_chip = initial_squad.active_chip
        players = projections.index
        positions = ('G', 'D', 'M', 'F')
        teams = projections['Team'].unique()
        gw_interval = list(range(next_gw, next_gw + self.horizon))

        # Initialise optimisation model
        prob = LpProblem('FPL_transfer_optimisation')

        # Initialise decision variables
        default_args = {'index': players, 'columns': gw_interval, 'column_type': 'gw', 'model': prob}
        lineup = DecisionMatrix.lp_variable('lineup', **default_args)
        bench_gk = DecisionMatrix.lp_variable('bench_gk', **default_args)
        bench_1 = DecisionMatrix.lp_variable('bench_1', **default_args)
        bench_2 = DecisionMatrix.lp_variable('bench_2', **default_args)
        bench_3 = DecisionMatrix.lp_variable('bench_3', **default_args)
        squad = lineup + bench_gk + bench_1 + bench_2 + bench_3
        squad[0] = players.isin(initial_players).astype(int)
        captain = DecisionMatrix.lp_variable('captain', **default_args)
        vice_captain = DecisionMatrix.lp_variable('vice_captain', **default_args)
        transfer_in = DecisionMatrix.lp_variable('transfer_in', **default_args)
        transfer_out = DecisionMatrix.lp_variable('transfer_out', **default_args)
        itb = DecisionSeries(data=[initial_itb], index=[next_gw - 1], model=prob)
        # itb is previous GW's itb + revenue from outgoing players + cost of incoming players
        for i, gw in enumerate(gw_interval):
            itb[gw] = itb[gw - 1] + (transfer_out[gw] * projections['SV']).sum() - \
                      (transfer_in[gw] * projections['BV']).sum() - self.budget_decay_rate

        # Add problem constraints to optimisation model
        prob += squad == squad.lag(1) + transfer_in - transfer_out  # New squad is previous squad plus transfers
        prob += lineup.sum() == 11  # Lineup contains 11 players
        prob += bench_gk.sum() == 1     # There is 1 bench GK;
        prob += bench_1.sum() == 1  # 1 1st bench slot;
        prob += bench_2.sum() == 1  # 1 2nd bench slot;
        prob += bench_3.sum() == 1  # 1 3rd bench slot;
        prob += captain.sum() == 1  # 1 Captain;
        prob += vice_captain.sum() == 1     # 1 vice-captain
        prob += captain <= lineup   # Captain must be in lineup
        prob += vice_captain <= lineup  # Vice-captain must be in lineup
        prob += captain + vice_captain <= 1     # Captain and vice-captain must be different players
        for position, limit in zip(positions, (2, 5, 5, 3)):
            prob += squad[projections['Pos'] == position].sum() == limit    # Set squad position structure
        for team in teams:
            prob += squad[projections['Team'] == team].sum() <= 3   # No more than 3 players from each team
        prob += bench_gk <= (projections['Pos'] == 'G')     # Bench GK must be a goalkeeper
        prob += (lineup * (projections['Pos'] == 'G')).sum() == 1   # There must be 1 goalkeeper in lineup
        prob += (lineup * (projections['Pos'] == 'D')).sum() >= 3   # There must be at least 3 defenders in lineup
        prob += itb >= 0    # The amount of money in the bank must be non-negative

        # Set up transfer logic
        transfer_args = {'index': gw_interval, 'column_type': 'gw', 'model': prob, 'cat': 'Integer'}
        aux = DecisionSeries.lp_variable('aux', **transfer_args)
        free_transfers = DecisionSeries(data=[initial_fts], index=[0], model=prob) + DecisionSeries.lp_variable(
            'free_transfers', **transfer_args)
        penalised_transfers = DecisionSeries.lp_variable('penalised_transfers', **transfer_args)
        transfer_counts = transfer_in.sum()
        frees_minus_transfers = free_transfers.lag(1) - transfer_counts
        lower_bound = aux * 15 - 14
        upper_bound = aux * 2
        prob += frees_minus_transfers >= lower_bound
        prob += frees_minus_transfers <= upper_bound
        prob += free_transfers == aux + 1
        prob += penalised_transfers >= -frees_minus_transfers
        prob += penalised_transfers >= 0

        ev_values = projections[[f'{gw}_Pts' for gw in gw_interval]]    # Restructure projections data for easier
        ev_values.columns = gw_interval                                 # manipulation
        objective = ((lineup + captain) * ev_values).sum()  # Objective function is sum of lineup and captain pts
        for loc, bench_slot in enumerate((bench_gk, bench_1, bench_2, bench_3)):
            objective += (bench_slot * ev_values).sum() * self.bench_weights[:, loc]    # Add bench weights to objective
        if force_transfers is None:
            objective -= penalised_transfers * 4    # Take away 4 points from each hit taken
            if force_chips is not None:
                self.force_chips = force_chips
                for gw in force_chips:
                    if force_chips[gw] == 'wildcard':
                        objective[gw] += penalised_transfers[gw] * 4    # Remove penalised points in wildcard week
        self.rolls = frees_minus_transfers + penalised_transfers

        if self.penalties is not None:
            for penalty, parameter in self.penalties.items():
                objective = penalty(objective, self, parameter)  # Take away external solver penalty functions

        prob.model += objective.sum()
        prob.solve()
        return Solution(lineup, bench_gk, bench_1, bench_2, bench_3, objective, transfer_in, transfer_out)


def ft_penalty(objective, solver, parameter):
    if solver.force_chips is not None:
        for gw in solver.force_chips:
            if solver.force_chips[gw] == 'wildcard':
                solver.rolls[gw] = 0
    return objective + solver.rolls * parameter


def time_decay(objective, solver, parameter):
    return objective * np.array([parameter**i for i in range(solver.horizon)])


class Solution:
    def __init__(self, lineup, bench_gk, bench_1, bench_2, bench_3, objective,
                 transfer_in, transfer_out, next_gw=1, season=2021):
        self.data = pd.read_csv(root_dir + f'/data/fplreview/{season}/GW{next_gw}.csv')    # Get current EV projections
        self.lineup = lineup
        self.bench_gk = bench_gk
        self.bench_1 = bench_1
        self.bench_2 = bench_2
        self.bench_3 = bench_3
        self.transfer_in = transfer_in
        self.transfer_out = transfer_out
        self.objective = objective

    def __str__(self):
        output = ''
        for gw in self.lineup.columns:
            players_in = self.data['Name'][
                [player for player in self.data.index if self.transfer_in[player, gw].varValue]].tolist()
            players_out = self.data['Name'][
                [player for player in self.data.index if self.transfer_out[player, gw].varValue]].tolist()
            players_in, players_out = [player for player in players_in if player not in players_out], \
                                      [player for player in players_out if player not in players_in]

            output += '-' * 40 + '\n' + 'Transfer out: ' + ', '.join(players_out) + '\n'
            output += 'Transfer in: ' + ', '.join(players_in) + '\n' + '-' * 40 + '\n'

            for position in ['G', 'D', 'M', 'F']:
                players = self.data[self.data['Pos'] == position]
                gw_lineup = sorted(
                    players['Name'][[player for player in players.index if self.lineup[player, gw].varValue]].tolist())
                output += '  '.join(gw_lineup).center(40, ' ') + '\n'
            bench = []
            for bench_slot in [self.bench_gk, self.bench_1, self.bench_2, self.bench_3]:
                bench.append(self.data['Name'][[
                    player for player in self.data.index if bench_slot[player, gw].varValue]].tolist()[0])
            output += '-' * 40 + '\n' + '  '.join(bench).center(40, ' ') + '\n' + '=' * 40 + '\n'
            output += f'GW{gw} Objective: {pulp.value(self.objective[gw])}' + '\n' * 3
        return output[:-3]

    def eval(self):
        pass


if __name__ == '__main__':
    ev = (get_fplreview(2021, 1))
    my_squad = Squad(players=[68, 375, 141, 70, 219, 274, 236, 232, 250, 140, 195, 114, 39, 77, 314])
    my_solver = Solver(horizon=8, penalties={ft_penalty: 0.75, time_decay: 0.84})
    print(my_solver.solve(ev, my_squad, force_chips={1: 'wildcard'}))
