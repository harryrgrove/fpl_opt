import operator
import itertools as it

from definitions import root_dir
import pandas as pd
import pulp
from prettytable import PrettyTable


class Solver:
    """
    Linear optimisation solver for Fantasy Premier League
    =====================================================
    Defines a structure for a linear FPL solver, with method to apply the solver to instances of teams
    """
    def __init__(self, horizon=8, bench_weights=None, vc_weight=0, decay=1, penalties=None, force_players=None,
                 force_transfers=None, linear_budget_decay=0, transfer_pattern=None, no_everton=False):
        """
        :param horizon: length of transfer time horizon, default/max=8
        :type horizon: int
        :param bench_weights: [bench_gk_weight, 1st_bench_weight, 2nd_bench_weight, 3rd_bench_weight]
        :type bench_weights: list
        :param vc_weight: additional weight of vice captain in objective function
        :type vc_weight: float
        :param decay: exponential time decay parameter; gw N has weight decay^(N-1) in objective function
        :type decay: float in [0:1]
        :param penalties: list of linear functions with float outputs for penalisation elements of transfer plans
        :type penalties: list
        :param force_players: {'include': [id, ...], 'exclude': [id, ...]}
        :type force_players: dict
        :param force_transfers: {gw: {'in': [id_1, id_2], 'out': [id_3, id_4]}
        :type force_transfers: dict
        :param transfer_pattern: [transfers_before_next_gw, transfers_before_next^2_gw, etc until horizon]
        :type transfer_pattern: list
        :param linear_budget_decay: amount of money lost from team value each week, to account for external price changes
        :type linear_budget_decay: float
        :param no_everton: True if no Everton players are wanted in the optimal solution
        :type no_everton: bool
        """
        self.horizon = horizon
        self.bench_weights = bench_weights
        if bench_weights is None:
            self.bench_weights = [0.02, 0.4, 0.15, 0.01]
        self.vc_weight = vc_weight
        self.decay = decay
        self.penalties = penalties
        if penalties is None:
            self.penalties = []
        self.force_players = force_players
        self.force_transfers = force_transfers
        self.transfer_pattern = transfer_pattern
        self.budget_decay = linear_budget_decay
        self.no_everton = no_everton

    def __repr__(self):
        x = PrettyTable()
        x.field_names = ["Parameter", "Parameter Value"]
        x.add_rows(
            [
                ["GW Horizon", self.horizon],
                ["Bench Weights", self.bench_weights],
                ["Vice-Captain Weight", self.vc_weight],
                ["Time Decay Rate", self.decay],
                ["Exclude Everton Players?", self.no_everton]
            ]
        )
        if self.penalties:
            x.add_row(["Penalties", ', '.join([p.__name__ for p in self.penalties])])
        if self.force_transfers:
            x.add_row(["Forced Transfers", self.force_transfers])
        return "===================\n FPL Linear Solver\n===================\n" + str(x)

    def solve(self, next_gw=1, initial_squad=None, initial_itb=0,
              initial_fts=1, chips=None, active_chip=None, season=2021):
        """
        Method to apply solver to a given squad at a given time, with initial constraints
        With inspiration from: https://github.com/sertalpbilal/FPL-Optimization-Tools/blob/main/src/multi_period.py

        :param next_gw: next gw
        :type next_gw: int
        :param initial_squad: list containing element ids of players in initial team
        :type initial_squad: list
        :param initial_itb: money currently in the bank, in millions
        :type initial_itb: float
        :param initial_fts: number of free transfers currently available
        :type initial_fts: int
        :param chips: {'bench_boost': gw_for_BB, 'wildcard': gw_for_WC, 'free_hit': gw_for_FH}
        :type chips: dict
        :param active_chip: chip being currently used by manager
        :type active_chip: str
        :param season: year of start of relevant season
        :type season: int
        :return: optimal transfer plan given parameters
        """
        self.horizon = min(self.horizon, 39 - next_gw)

        # Initialise projection/ownership data
        if initial_squad is None:
            initial_squad = \
                [68, 375, 141, 70, 219, 274, 236, 232, 250, 140, 195, 114, 39, 77, 314]     # Set up default GW1 team
        data = pd.read_csv(root_dir + f'/data/fplreview/{season}/GW{next_gw}.csv')    # Get current EV projections

        # Useful reference sets
        players = data.index.tolist()
        positions = ['G', 'D', 'M', 'F']
        teams = data['Team'].unique()
        gw_interval = list(range(next_gw, next_gw + self.horizon))
        all_gws = [next_gw - 1] + gw_interval

        class DecisionSeries:
            """
            Data structure for easier manipulation of dict-like set of decision variables in linear programming
            Structured like pd.Series with support for LpVariable objects, using elementwise operations
            """
            def __init__(self, var_name=None, column_name=None, index=None, from_data=None,
                         column_type='gw', cat='Binary'):
                """
                :param var_name: Name of variable type e.g. "bench"
                :param column_name: Instance of column e.g. [gw] 1
                :param index: Index set for variables, e.g. players
                :param from_data: Converts dict containing decision variable data to DecisionSeries
                :param column_type: Type of column e.g. gw
                :param cat: Decision variable type
                """
                if from_data is None:
                    if index is None:
                        self.index = players
                    else:
                        self.index = index
                    if var_name is not None and column_name is not None:
                        self.data = pulp.LpVariable.dicts(f'{var_name}_{column_type}{column_name}', self.index, cat=cat)
                    else:
                        raise Exception('If new instance of decision variable, specify variable/column names')
                else:
                    self.index = list(from_data.keys())
                    self.data = from_data

            def __add__(self, other):
                if hasattr(other, '__getitem__'):
                    return DecisionSeries(from_data={i: self[i] + other[i] for i in self.index})
                else:
                    return DecisionSeries(from_data={i: self[i] + other for i in self.index})

            def __sub__(self, other):
                if hasattr(other, '__getitem__'):
                    return DecisionSeries(from_data={i: self[i] - other[i] for i in self.index})
                else:
                    return DecisionSeries(from_data={i: self[i] - other for i in self.index})

            def __mul__(self, other):
                if hasattr(other, '__getitem__'):
                    return DecisionSeries(from_data={i: self[i] * other[i] for i in self.index})
                else:
                    return DecisionSeries(from_data={i: self[i] * other for i in self.index})

            __rmul__ = __mul__

            def __neg__(self):
                return self * -1

            def __iter__(self):
                for value in list(self.data.values()):
                    yield value

            def __getitem__(self, idx):
                return self.data[idx]

            def __setitem__(self, key, value):
                self.data[key] = value
                self.index = sorted(self.data.keys())

            def __str__(self):
                keys = list(self.data.keys())
                values = [str(val) for val in self.data.values()]
                return str(pd.Series(index=keys, data=values).sort_index())

            def sum(self):
                return pulp.lpSum(self.data)

            def add_constraint(self, model, relation, other):
                """
                :param model: LpProblem object
                :param relation: Constraint operator: one from (operator.eq, operator.ge, operator.le, '==', '>=', '<=')
                :param other: Object for decision_series to be constrained against
                :return:
                """
                if relation in ('==', '>=', '<='):
                    relation = (operator.eq, operator.ge, operator.le)[('==', '>=', '<=').index(relation)]
                elif relation not in (operator.eq, operator.ge, operator.le):
                    raise Exception('Invalid relation operator')
                if hasattr(other, '__getitem__'):
                    for i in self.index:
                        model += relation(self[i], other[i])
                else:
                    for i in self.index:
                        model += relation(self[i], other)

            def value(self):
                return DecisionSeries(from_data={i: pulp.value(self[i]) for i in self.index})

        class DecisionMatrix:
            """
            Structure for related instances of DecisionSeries, similar to pd.DataFrame
            """
            def __init__(self, var_name=None, columns=None, index=None, from_data=None, column_type='gw', cat='Binary'):
                """
                :param var_name: Name of variable type e.g. "bench"
                :param columns: Contains names of columns e.g. gw_interval
                :param index: Index set for variables, e.g. players
                :param from_data: dict {col: DecisionSeries, ...}}
                :param column_type: Type of column e.g. gw
                :param cat: Decision variable type
                """
                if index is None:
                    self.index = players
                else:
                    self.index = index
                if columns is None:
                    self.columns = gw_interval
                else:
                    self.columns = columns
                if from_data is None:
                    if var_name is not None:
                        self.data = {column: DecisionSeries(
                            var_name, column, self.index, column_type=column_type, cat=cat) for column in self.columns}
                    else:
                        raise Exception('If new instance of decision variable, specify variable name')
                else:
                    self.data = from_data

            def __add__(self, other):
                if type(other) in [pd.DataFrame, DecisionMatrix]:
                    return DecisionMatrix(from_data={column: self[column] + other[column] for column in self.columns})
                else:
                    return DecisionMatrix(from_data={column: self[column] + other for column in self.columns})

            def __sub__(self, other):
                if type(other) in [pd.DataFrame, DecisionMatrix]:
                    return DecisionMatrix(from_data={column: self[column] - other[column] for column in self.columns})
                else:
                    return DecisionMatrix(from_data={column: self[column] - other for column in self.columns})

            def __mul__(self, other):
                if type(other) in [pd.DataFrame, DecisionMatrix]:
                    return DecisionMatrix(from_data={column: self[column] * other[column] for column in self.columns})
                else:
                    return DecisionMatrix(from_data={column: self[column] * other for column in self.columns})

            __rmul__ = __mul__

            def __neg__(self):
                return self * -1

            def __getitem__(self, pos):
                if type(pos) == int:
                    return self.data[pos]
                elif type(pos) == tuple:
                    if len(pos) == 2:
                        idx, column = pos
                        return self.data[column][idx]
                raise Exception('Invalid index')

            def __setitem__(self, key, values):
                self.data[key] = DecisionSeries(from_data={self.index[i]: values[i] for i in range(len(self.index))})
                self.columns = sorted(self.data.keys())

            def __str__(self):
                df = pd.DataFrame(index=self.index)
                for column in self.columns:
                    df[column] = [str(i) for i in list(self[column].data.values())]
                return str(df)

            def sum(self):
                return DecisionSeries(
                    from_data={column: self.data[column].sum() for column in self.columns})

            def add_constraint(self, model, relation, other):
                """
                :param model: LpProblem object
                :param relation: Constraint operator: one from (operator.eq, operator.ge, operator.le, '==', '>=', '<=')
                :param other: Object for decision_series to be constrained against
                :return:
                """
                if relation in ('==', '>=', '<='):
                    relation = (operator.eq, operator.ge, operator.le)[('==', '>=', '<=').index(relation)]
                elif relation not in (operator.eq, operator.ge, operator.le):
                    raise Exception('Invalid relation operator')
                if hasattr(other, '__getitem__'):
                    if type(other) == DecisionMatrix:
                        for i, column in it.product(self.index, self.columns):
                            model += relation(self[column][i], other[column][i])
                    else:
                        for i, column in it.product(self.index, self.columns):
                            model += relation(self[column][i], other[i])
                else:
                    for i, column in it.product(self.index, self.columns):
                        model += relation(self[column][i], other)

        prob = pulp.LpProblem('fpl_opt', pulp.LpMaximize)     # Initialise optimisation model

        # Initialise decision variables
        lineup = DecisionMatrix('lineup')
        bench_gk = DecisionMatrix('bench_gk')
        bench_1 = DecisionMatrix('bench_1')
        bench_2 = DecisionMatrix('bench_2')
        bench_3 = DecisionMatrix('bench_3')
        bench = bench_gk + bench_1 + bench_2 + bench_3
        squad = lineup + bench
        captain = DecisionMatrix('captain')
        vice_captain = DecisionMatrix('vice_captain')
        transfer_in = DecisionMatrix('transfer_in')
        transfer_out = DecisionMatrix('transfer_out')

        # Add problem constraints to optimisation model
        squad[0] = {player: player in initial_squad for player in players}
        for gw in gw_interval:
            squad[gw].add_constraint(prob, '==', squad[gw - 1] + transfer_in[gw] - transfer_out[gw])
        squad.sum().add_constraint(prob, '==', 15)  # Squad is composed of 15 players
        lineup.sum().add_constraint(prob, '==', 11)     # Lineup is composed of 11 players
        bench_gk.sum().add_constraint(prob, '==', 1)     # Only 1 GK on the bench
        bench_1.sum().add_constraint(prob, '==', 1)     # 1 player in 1st bench slot
        bench_2.sum().add_constraint(prob, '==', 1)     # 1 player in 2nd bench slot
        bench_3.sum().add_constraint(prob, '==', 1)  # 1 player in 3st bench slot
        for position, limit in zip(positions, (2, 5, 5, 3)):    # Right number of players of each position in squad
            (squad * (data['Pos'] == position)).sum().add_constraint(prob, '==', limit)
        for team in teams:
            (squad * (data['Team'] == team)).sum().add_constraint(prob, '<=', 3)
        (bench_gk * (data['Pos'] == 'G')).sum().add_constraint(prob, '==', 1)   # Bench GK must be a GK
        (lineup * (data['Pos'] == 'G')).sum().add_constraint(prob, '==', 1)     # Lineup has 1 GK
        (lineup * (data['Pos'] == 'D')).sum().add_constraint(prob, '>=', 3)     # Lineup has at least 3 DFs
        captain.sum().add_constraint(prob, '==', 1)     # Only one captain in lineup
        vice_captain.sum().add_constraint(prob, '==', 1)    # Only one vice-captain in lineup
        captain.add_constraint(prob, '<=', lineup)  # Captain must be in lineup
        vice_captain.add_constraint(prob, '<=', lineup)  # Vice-captain must be in lineup

        in_the_bank = {0: initial_itb}  # Initialise itb variable
        for i, gw in enumerate(gw_interval):    # itb += sold_value - bought_value
            in_the_bank[gw] = in_the_bank[gw - 1] + (transfer_out[gw] * data['SV']).sum() - \
                              (transfer_in[gw] * data['BV']).sum() - round(self.budget_decay * i, 4)

        in_the_bank = DecisionSeries(from_data=in_the_bank)     # Convert in_the_bank to DecisionSeries
        in_the_bank.add_constraint(prob, '>=', 0)   # There must be at least 0 in the bank

        constrained_gws = list(gw_interval)   # List of gws for which transfer constraints are relevant
        if chips:
            if 'wildcard' in chips:
                constrained_gws.remove(chips['wildcard'])   # Wildcard gw has no transfer constraints
            if 'bench_boost' in chips:
                pass
            if 'free_hit' in chips:
                pass

        if self.transfer_pattern:
            penalised_transfers = {gw: 0 for gw in gw_interval}
            for i, gw in enumerate(gw_interval):
                if gw in constrained_gws:   # For unconstrained gws
                    prob += transfer_in[gw].sum() <= self.transfer_pattern[i]   # Transfers must be less than spec
        else:   # Full transfer logic implementation
            aux = DecisionSeries('aux', column_name='', index=gw_interval, column_type='', cat='Integer')
            free_transfers = DecisionSeries(
                'free_transfers', column_name='', index=gw_interval, column_type='', cat='Integer')
            free_transfers[0] = initial_fts
            penalised_transfers = DecisionSeries(
                'penalised_transfers', column_name='', index=gw_interval, column_type='', cat='Integer')
            transfer_counts = transfer_in.sum()
            frees_minus_transfers = DecisionSeries(
                from_data={gw: free_transfers[gw - 1] - transfer_counts[gw] for gw in gw_interval}, cat='Integer')
            transfer_diffs = -frees_minus_transfers
            upper_bound = 2 * aux
            lower_bound = 15 * aux - 14
            frees_minus_transfers.add_constraint(prob, '<=', upper_bound)
            frees_minus_transfers.add_constraint(prob, '>=', lower_bound)
            (aux + 1).add_constraint(prob, '==', free_transfers)
            penalised_transfers.add_constraint(prob, '>=', transfer_diffs)
            penalised_transfers.add_constraint(prob, '>=', 0)
            print(frees_minus_transfers)

        projections = pd.DataFrame(data[[f'{gw}_Pts' for gw in gw_interval]])
        projections.columns = gw_interval
        objective = (lineup * projections).sum() + (captain * projections).sum() +\
                    (vice_captain * projections * self.vc_weight).sum()
        obj_var = DecisionSeries('obj_var', column_name='', index=gw_interval, column_type='', cat='Continuous')
        obj_var.add_constraint(prob, '==', objective)
        for bench_slot, weight in zip([bench_gk, bench_1, bench_2, bench_3], self.bench_weights):
            objective += (bench_slot * projections * weight).sum()  # Add bench weight to objective
        objective -= penalised_transfers * 4    # Add hits
        for gw in set(gw_interval).difference(set(constrained_gws)):
            objective[gw] += penalised_transfers[gw] * 4    # Remove hits for GWs where transfers aren't constrained
        objective *= {gw: self.decay**k for k, gw in enumerate(gw_interval)}    # Apply time decay
        prob += objective.sum()
        prob.solve()    # Solve optimisation problem
        print(ft_penalty(gw_interval, constrained_gws, frees_minus_transfers, 2))
        print(objective)
        print(objective * ft_penalty(gw_interval, constrained_gws, frees_minus_transfers, 2))
        return Solution(lineup, bench_gk, bench_1, bench_2, bench_3, obj_var, transfer_in, transfer_out)


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
            output += f'GW{gw} Objective: {self.objective[gw].varValue}' + '\n' * 3
        return output[:-3]

    def eval(self):
        pass


def ft_penalty(gw_interval, constrained_gws, frees_minus_transfers, param=0):
    print(frees_minus_transfers.value())
    output = {gw: frees_minus_transfers.value()[gw] * param for gw in constrained_gws}
    for gw in set(gw_interval).difference(set(constrained_gws)):
        output[gw] = 0
    return output


def roll_incentive(gw_interval, constrained_gws, frees_minus_transfers, param=0):
    if min(gw_interval) in constrained_gws:
        return 0


s = Solver(decay=0.84, horizon=5, penalties=[ft_penalty])
print(s)

solution = s.solve(chips={'wildcard': 1})
print(solution)
