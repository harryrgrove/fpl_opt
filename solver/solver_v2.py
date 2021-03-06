from solver.lp_tools import LpProblem, DecisionSeries, DecisionMatrix
from solver.fpl_tools import Projections, Squad
from definitions import root_dir
import numpy as np
from typing import List, Callable, Dict, Iterable
import pandas as pd
import pulp


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
            vc_weight: float = 0.05,
            penalties: Dict[Callable, float] = None,
            budget_decay_rate: float = 0,
            transfer_pattern: List[int] = None,
            million_value=0.45,
            exclude_everton: bool = False,
    ):
        """
        :param horizon: Number of future GWs included in optimisation, max=8
        :param bench_weights: [bench_gk_weight, 1st_bench_weight, 2nd_bench_weight, 3rd_bench_weight]
        :param vc_weight: Extra weight of vice captain in objective function
        :param penalties: Dict of penalty functions taking self, current objective and parameter
        :param budget_decay_rate: Linear decay parameter; budget reduces by budget_decay_rate each future GW
        :param transfer_pattern: Manual override for transfer logic, nth element is transfers in n GWs from now
        :param million_value: User-defined value of £1 million in FPL, used to factor in price changes. "solve" solves.
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
        self.penalties = penalties
        self.budget_decay_rate = budget_decay_rate
        self.transfer_pattern = transfer_pattern
        if million_value == 'solve':
            # Investigate value of 1 million in current FPL game state
            pass
        else:
            pass
        self.million_value = million_value
        self.exclude_everton = exclude_everton

        self.rolls = None
        self.force_chips = None

    def solve(
            self,
            projections: Projections,
            initial_squad: Squad,
            next_gw: int = None,
            force_chips: Dict[int, str] = None,
            force_players: Dict[str, list] = None,
            force_transfers: Dict[int, dict] = None,
            price_changes: Dict[str, Iterable] = None,
            time_limit: float = None,
            optimizer: type = pulp.GUROBI,
            message: bool = True
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
        :param price_changes:
        :param time_limit: Set time limit for optimisation (in seconds)
        :param optimizer: Which optimiser to use
        :param message: if False, suppress optimisation output:
        :return: class Solution containing transfer and team selection data for optimal solution
        """
        if next_gw is None:
            next_gw = sorted([int(column.split('_')[0]) for column in projections.columns if column.endswith('Pts')])[0]
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
        squad = DecisionMatrix.lp_variable('squad', **default_args)
        prob += squad == lineup + bench_gk + bench_1 + bench_2 + bench_3
        squad[next_gw - 1] = pd.Series(squad.index).isin(initial_players).astype(int)
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
        prob += squad.drop(next_gw - 1, axis=1) <= 1  # Each player can only appear in the squad once
        prob += lineup.sum() == 11  # Lineup contains 11 players
        prob += bench_gk.sum() == 1     # There is 1 bench GK;
        prob += bench_1.sum() == 1  # 1 1st bench slot;
        prob += bench_2.sum() == 1  # 1 2nd bench slot;
        prob += bench_3.sum() == 1  # 1 3rd bench slot;
        prob += captain.sum() == 1  # 1 Captain;
        prob += transfer_out.sum() == transfer_in.sum()     # Transfers in must be same as transfers out

        prob += vice_captain.sum() == 1     # 1 vice-captain
        prob += captain <= lineup   # Captain must be in lineup
        prob += vice_captain <= lineup  # Vice-captain must be in lineup
        prob += captain + vice_captain <= 1     # Captain and vice-captain must be different players
        for position, limit in zip(positions, (2, 5, 5, 3)):
            prob += squad[projections['Pos'] == position].sum() == limit    # Set squad position structure
        for team in teams:
            prob += squad[projections['Team'] == team].sum() <= 3   # No more than 3 players from each team
        if self.exclude_everton:
            prob += squad[projections['Team'] == 'Everton'].sum() == 0  # Option to exclude Everton players
        prob += bench_gk <= (projections['Pos'] == 'G')     # Bench GK must be a goalkeeper
        prob += (lineup * (projections['Pos'] == 'G')).sum() == 1   # There must be 1 goalkeeper in lineup
        prob += (lineup * (projections['Pos'] == 'D')).sum() >= 3   # There must be at least 3 defenders in lineup
        prob += itb[[False] + [True] * self.horizon] >= 0   # The itb amount must be non-negative for future GWs

        # Set up transfer logic
        transfer_args = {'index': gw_interval, 'column_type': 'gw', 'model': prob, 'cat': 'Integer'}
        aux = DecisionSeries.lp_variable('aux', **transfer_args)
        free_transfers = DecisionSeries(data=[initial_fts], index=[next_gw - 1], model=prob) + DecisionSeries.\
            lp_variable('free_transfers', **transfer_args)
        penalised_transfers = DecisionSeries.lp_variable('penalised_transfers', **transfer_args)
        transfer_counts = transfer_in.sum()
        frees_minus_transfers = free_transfers.lag(1) - transfer_counts
        lower_bound = aux * 15 - 14
        upper_bound = aux * 2
        if initial_fts > 1:
            prob += transfer_counts[next_gw] >= 1
        prob += frees_minus_transfers >= lower_bound
        prob += frees_minus_transfers <= upper_bound
        prob += free_transfers == aux + 1
        # penalised_transfers is max(transfers - frees, 0)
        prob += penalised_transfers >= -frees_minus_transfers
        prob += penalised_transfers >= 0

        ev_values = projections[[f'{gw}_Pts' for gw in gw_interval]]    # Restructure projections data for easier
        ev_values.columns = gw_interval                                 # manipulation
        objective = ((lineup + captain) * ev_values).sum()  # Objective function is sum of lineup and captain pts
        objective += (vice_captain * self.vc_weight * ev_values).sum()  # Add vice-captain weight
        for loc, bench_slot in enumerate((bench_gk, bench_1, bench_2, bench_3)):
            objective += (bench_slot * ev_values).sum() * self.bench_weights[:, loc]    # Add bench weights to objective
        if force_transfers is None:
            objective -= penalised_transfers * 4    # Take away 4 points from each hit taken
            if force_chips is not None:
                self.force_chips = force_chips
                for gw in force_chips:
                    if force_chips[gw] == 'wildcard':
                        objective[gw] += penalised_transfers[gw] * 4    # Remove penalised points in wildcard week

        if force_players is not None:
            for player in force_players['include']:
                prob += squad.T[player].drop(next_gw - 1) == 1
            for player in force_players['exclude']:
                prob += squad.T[player].drop(next_gw - 1) == 0
            if 'include_for_gw' in force_players:
                for gw in force_players['include_for_gw']:
                    try:
                        prob += squad[force_players['include_for_gw'][gw], gw] == 1
                    except ValueError:
                        pass
            if 'exclude_for_gw' in force_players:
                for gw in force_players['exclude_for_gw']:
                    try:
                        prob += squad[force_players['exclude_for_gw'][gw], gw] == 0
                    except ValueError:
                        pass
        self.rolls = frees_minus_transfers + penalised_transfers
        prob += self.rolls <= 2

        if self.penalties is not None:
            if time_decay in self.penalties:
                self.penalties[time_decay] = self.penalties.pop(time_decay)     # Apply time decay after other penalties
            for penalty, parameter in self.penalties.items():
                objective = penalty(objective, self, parameter)  # Apply external penalty functions

        # Apply price change EV
        if price_changes is not None:
            gws_remaining = 38 - next_gw + 1
            for player in price_changes['rise']:
                objective[next_gw] += self.million_value / 30 * squad[player, next_gw] * gws_remaining
            for player in price_changes['drop']:
                objective[next_gw] -= self.million_value / 10 * squad[player, next_gw] * gws_remaining

        prob.model += objective.sum()
        prob.solve(time_limit=time_limit, optimizer=optimizer, message=message)

        return Solution(lineup, bench_gk, bench_1, bench_2, bench_3, captain, vice_captain, objective, transfer_in,
                        transfer_out, itb, projections, free_transfers, penalised_transfers, force_chips)


class ObjectivePenalty:     # TODO
    def __init__(self, penalty_func, parameter):
        pass

    def apply(self, objective):
        pass


def ft_penalty(objective, solver, parameter):
    if parameter == 0:  # Unsolved glitch occurs when ft_penalty = 0
        parameter = 0.01
    if solver.force_chips is not None:
        for gw in solver.force_chips:
            if solver.force_chips[gw] == 'wildcard':
                solver.rolls[gw] = 0
    return objective + solver.rolls * parameter


def time_decay(objective, solver, parameter):
    return objective * np.array([parameter**i for i in range(solver.horizon)])


class Solution:
    def __init__(self, lineup, bench_gk, bench_1, bench_2, bench_3, captain, vice_captain, objective,
                 transfer_in, transfer_out, itb, projections, free_transfers, penalised_transfers, chips):
        self.data = projections    # Get current EV projections
        self.lineup = lineup
        self.bench_gk = bench_gk
        self.bench_1 = bench_1
        self.bench_2 = bench_2
        self.bench_3 = bench_3
        self.captain = captain
        self.vice_captain = vice_captain
        self.transfer_in = transfer_in
        self.transfer_out = transfer_out
        self.itb = itb
        self.objective = objective
        self.free_transfers = free_transfers
        self.penalised_transfers = penalised_transfers
        self.chips = chips
        self.bench_weights = None

    def __str__(self):
        output = ''
        lineup, bench_gk, bench_1, bench_2, bench_3, captain, vice_captain, transfer_in, transfer_out = \
            [variable.get_value() for variable in (
                self.lineup, self.bench_gk, self.bench_1, self.bench_2, self.bench_3, self.captain, self.vice_captain,
                self.transfer_in, self.transfer_out)]
        for gw in self.lineup.columns:
            players_in = [self.data.loc[i, 'Name'] for i in transfer_in[gw][pd.Series(transfer_in[gw]) >= 0.5].index]
            players_out = [self.data.loc[i, 'Name'] for i in transfer_in[gw][pd.Series(transfer_out[gw]) >= 0.5].index]
            players_in, players_out = [player for player in players_in if player not in players_out], \
                                      [player for player in players_out if player not in players_in]
            output += '-' * 60 + '\n' + 'Transfer out: ' + ', '.join(players_out) + '\n'
            output += 'Transfer in: ' + ', '.join(players_in) + '\n' + '-' * 60 + '\n'
            gw_lineup = lineup[gw][pd.Series(lineup[gw]) >= 0.5].index
            for position in ('G', 'D', 'M', 'F'):
                pos_players = [self.data.loc[i, 'Name'] + ' (C)' * int(captain[i, gw]) + ' (VC)' *
                               int(vice_captain[i, gw]) for i in sorted(gw_lineup, key=lambda i: self.data.loc[
                                i, f'{gw}_Pts'], reverse=True) if self.data.loc[i, 'Pos'] == position]
                output += '  '.join(pos_players).center(60, ' ') + '\n'
            gw_bench_gk = self.data.loc[bench_gk[gw][pd.Series(bench_gk[gw]) >= 0.5].index[0], 'Name']
            gw_bench_1 = self.data.loc[bench_1[gw][pd.Series(bench_1[gw]) >= 0.5].index[0], 'Name']
            gw_bench_2 = self.data.loc[bench_2[gw][pd.Series(bench_2[gw]) >= 0.5].index[0], 'Name']
            gw_bench_3 = self.data.loc[bench_3[gw][pd.Series(bench_3[gw]) >= 0.5].index[0], 'Name']
            bench = [gw_bench_gk, gw_bench_1, gw_bench_2, gw_bench_3]
            output += '-' * 60 + '\n' + '  '.join(bench).center(60, ' ') + '\n' + '=' * 60 + '\n'
            output += f'GW{gw} Objective: {pulp.value(self.objective[gw])}' + '\n' * 3
        return output[:-3]

    def eval(self, gws_from_now=0, bench_weights=None, vc_weight=0):
        if bench_weights is None:
            bench_weights = np.array([0.001, 0.15, 0.01, 0.001])
        gw = self.lineup.columns[gws_from_now]
        ev_sum = 0
        for variable in (self.lineup, self.captain):
            ev_sum += (variable[gw].get_value() * self.data[f'{gw}_Pts']).sum()
        for bench_slot, weight in zip([self.bench_gk, self.bench_1, self.bench_2, self.bench_3], bench_weights):
            ev_sum += (bench_slot[gw].get_value() * self.data[f'{gw}_Pts']).sum() * weight
        ev_sum += (self.vice_captain[gw].get_value() * self.data[f'{gw}_Pts']).sum() * vc_weight

        penalised_transfers = self.penalised_transfers.get_value()
        if self.chips is not None:
            for gw in self.chips:
                if self.chips[gw] == 'wildcard':
                    penalised_transfers[gw] = 0
        ev_sum -= penalised_transfers.sum() * 4

        return ev_sum


def main():
    gw = 7
    ev = Projections.get_fplreview(2021, gw)
    my_squad = Squad.from_id(1433, gw - 1)
    print(my_squad)

    solver = Solver(
        horizon=6,
        penalties={
            ft_penalty: 1.2,
            time_decay: 0.84
        },
        million_value=0.45
    )
    solution = solver.solve(
        projections=ev,
        initial_squad=my_squad,
        force_players={'include': [], 'exclude': [ev.get_player('Alonso')], 'include_for_gw': {}, 'exclude_for_gw': {}},
        force_chips={8: 'wildcard'},
        price_changes={'rise': [ev.get_player('Rudiger')], 'drop': [ev.get_player('Shaw')]}
    )
    print(solution)


if __name__ == '__main__':
    main()
