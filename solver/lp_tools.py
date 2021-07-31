import operator
import numpy as np
import pulp
import pandas as pd


class LpProblem:
    def __init__(self, name='NoName', sense=pulp.LpMaximize):
        self.model = pulp.LpProblem(name, sense)

    def __iadd__(self, other):
        return self

    def __str__(self):
        return str(self.model)

    def solve(self):
        return self.model.solve()


class DecisionSeries:
    """
    pd.Series-like object with wide support for pulp.LpVariable data
    """
    def __init__(self, data=None, index=None, model=None):
        """
        :param data: dict, DecisionSeries or array-like
            Contains data stored in DecisionSeries. If data is a dict, argument order is maintained.
        :param index: array-like
            Contains DecisionSeries index.
        :param model: pulp.LpProblem:
            pulp.LpProblem, model with which series is associated
        """

        if data is not None:
            if type(data) == dict:
                self.index, self.values = np.array(list(data.keys())), np.array(list(data.values()))
            elif type(data) in (DecisionSeries, pd.Series):
                self.index, self.values = data.index, data.values
            elif type(data) in (list, np.ndarray):
                self.index, self.values = range(len(data)), np.array(data)
            else:
                raise Exception(f"Data type '{type(data)}' is not valid")
        else:
            self.index, self.values = np.array([]), np.array([])

        if index is not None:
            self.index = np.array(index)

        if model is not None:
            self.model = model
        else:
            self.model = None

        if len(self.index) != len(self.values):
            raise Exception(f'DecisionArray index (len={len(self.index)}) and values'
                            f' (len={len(self.values)}) are not the same length')

    @classmethod
    def lp_variable(cls, name, index, column_type=None, column_instance='', low_bound=None, up_bound=None, cat='Binary',
                    model=None):
        if column_type is not None:
            return DecisionSeries(
                pulp.LpVariable.dict(f'{name}_{column_type}{column_instance}', index, low_bound, up_bound, cat),
                model=model)
        else:
            return DecisionSeries(pulp.LpVariable.dict(f'{name}', index, low_bound, up_bound, cat), model=model)

    def operation(self, other, op):
        """
        Operands of different sizes/indexes are supported, unmatched values are 0
        Operand with type list and np.array are unstable and not not recommended
        """
        if hasattr(other, '__getitem__'):
            if type(other) in (DecisionSeries, pd.Series):
                data = {idx: op(self[idx], other[idx]) for idx in set(self.index).intersection(set(other.index))} | {
                    idx: op(self[idx], 0) for idx in set(self.index) - set(other.index)} | {
                    idx: op(0, other[idx]) for idx in set(other.index) - set(self.index)}
            elif type(other) == dict:
                data = {idx: op(self[idx], other[idx]) for idx in set(self.index).intersection(set(other.keys()))} | {
                    idx: op(self[idx], 0) for idx in set(self.index) - set(other.keys())} | {
                    idx: op(0, other[idx]) for idx in set(other.keys()) - set(self.index)}
            elif type(other) in (list, np.array):
                data = {self.index[idx]: op(self.values[idx], other[idx]) for idx in range(min(len(self), len(other)))}\
                       | {self.index[idx]: op(self.values[idx], 0) for idx in range(len(other), len(self))} | {
                    idx + max(self.index) - len(self) + 1: op(0, other[idx]) for idx in range(len(self), len(other))}
            else:
                raise TypeError(f"unsupported operand type(s) for +: 'DecisionSeries' and '{type(other)}'")
            return DecisionSeries(dict(sorted(data.items())), model=self.model)
        else:
            return DecisionSeries({idx: op(self[idx], other) for idx in self.index}, model=self.model)

    def __add__(self, other):
        return self.operation(other, operator.add)

    __radd__ = __add__

    def __sub__(self, other):
        return self.operation(other, operator.sub)

    def __mul__(self, other):
        return self.operation(other, operator.mul)

    __rmul__ = __mul__

    def __neg__(self):
        return DecisionSeries(-self.values, self.index, self.model)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        try:
            return self.values[(list(self.index).index(item))]
        except ValueError:
            raise ValueError(f'{item} not in index')

    def __setitem__(self, key, value):
        self.values[(list(self.index).index(key))] = value

    def __iter__(self):
        for value in list(self.values):
            yield value

    def __str__(self):
        return str(pd.Series(data=[str(value) for value in self.values], index=self.index, dtype='object'))

    def sum(self):
        return pulp.lpSum(self.values)

    def add_constraint(self, model, relation, other):
        """
        :param model: pulp.LpProblem, model with which series is associated
        :param relation: in (operator.eq, operator.ge, operator.le, '==', '>=', '<=')
        :param other: DecisionSeries, pd.Series or dict
        """
        model = model.model
        if relation in ('==', '>=', '<='):
            relation = (operator.eq, operator.ge, operator.le)[('==', '>=', '<=').index(relation)]
        elif relation not in (operator.eq, operator.ge, operator.le):
            raise Exception('Invalid relation operator')
        if hasattr(other, '__getitem__'):
            if type(other) in [pd.Series, DecisionSeries]:
                for idx in set(self.index).intersection(set(other.index)):
                    model += relation(self[idx], other[idx])
            elif type(other) == dict:
                for idx in set(self.index).intersection(set(other.values())):
                    model += relation(self[idx], other[idx])
        else:
            for i in self.index:
                model += relation(self[i], other)

    def add_constraint_simple(self, relation, other):
        if not self.model:
            raise Exception('Model is not defined')
        return self.add_constraint(self.model, relation, other)

    def __ge__(self, other):
        return self.add_constraint_simple(operator.ge, other)

    def __le__(self, other):
        return self.add_constraint_simple(operator.le, other)

    def __eq__(self, other):
        return self.add_constraint_simple(operator.eq, other)

    def get_value(self):
        return DecisionSeries([pulp.value(item) for item in self.values], self.index, self.model)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class DecisionMatrix:
    """
    pd.DataFrame-like 2D collection of DecisionSeries objects
    """
    def __init__(self, data=None, index=None, columns=None, model=None):
        """
        :param data: np.ndarray,, Iterable, dict, or DecisionMatrix
        :param index: array-like
        :param columns: array-like
        """
        if data is not None:
            if type(data) == dict:
                self.columns = np.array(list(data.keys()))
                self.index = np.array(range(len(data[self.columns[0]])))
                self.values = np.array([DecisionSeries(data[column]).values for column in data]).T
            elif type(data) in (DecisionMatrix, pd.DataFrame):
                self.index, self.columns, self.values = data.index, data.columns, data.values
            elif type(data) in (list, np.ndarray):
                self.index, self.columns = [np.array(range(dim)) for dim in data.shape]
                self.values = np.array(data)
            else:
                raise Exception(f"Data type '{type(data)}' is not valid")
        else:
            self.values, self.index, self.columns = [np.array([])] * 3

        if index is not None:
            self.index = np.array(index)

        if columns is not None:
            self.columns = np.array(columns)

        self.model = model

    @classmethod
    def lp_variable(cls, name, index, columns, column_type='', cat='Binary', low_bound=None, up_bound=None, model=None):
        return DecisionMatrix({column: pulp.LpVariable.dict(f'{name}_{column_type}{column}', index, low_bound, up_bound,
                                                            cat) for column in columns}, index, columns, model)

    def operation(self, other, op):
        new = DecisionMatrix(self, model=self.model)
        if type(other) in (DecisionMatrix, pd.DataFrame):
            for column in set(new.columns).intersection(other.columns):
                new[column] = op(DecisionSeries(new[column]), DecisionSeries(other[column]))
        else:
            for column in new.columns:
                new[column] = op(DecisionSeries(new[column]), other)
        return new

    def __add__(self, other):
        return self.operation(other, operator.add)

    __radd__ = __add__

    def __sub__(self, other):
        return self.operation(other, operator.sub)

    def __mul__(self, other):
        return self.operation(other, operator.mul)

    __rmul__ = __mul__

    def __getitem__(self, item):
        if type(item) == tuple:
            index, col = item
            return self[col][index]
        else:
            try:
                column_loc = list(self.columns).index(item)
                return DecisionSeries([row[column_loc] for row in self.values], self.index, self.model)
            except ValueError:
                raise ValueError(f'{item} not in columns')

    def __setitem__(self, key, value):
        if type(key) == tuple:
            index, col = key
            self[col][index] = value
        elif key in self.columns:
            column_loc = list(self.columns).index(key)
            self.values[:, column_loc] = value
        else:
            self.columns = np.append(self.columns, key)
            if not hasattr(value, '__getitem__'):
                value = np.repeat(value, len(self.index))
            self.values = np.append(self.values, np.array(value)[:, None], axis=1)

    def __str__(self):
        return str(pd.DataFrame(data=self.values.astype(str), index=self.index, columns=self.columns))

    def sum(self):
        return DecisionSeries({column: self[column].sum() for column in self.columns}, model=self.model)

    def lag(self, value):
        return DecisionMatrix(self.values, self.index, self.columns + value, self.model)

    def add_constraint(self, model, relation, other):
        if type(other) in (DecisionMatrix, pd.DataFrame):
            for column in set(self.columns).intersection(set(other.columns)):
                self[column].add_constraint(model, relation, other[column])
        else:
            for column in self.columns:
                self[column].add_constraint(model, relation, other)

    def add_constraint_simple(self, relation, other):
        if self.model is None:
            raise Exception('model is not defined')
        return self.add_constraint(self.model, relation, other)

    def __ge__(self, other):
        return self.add_constraint_simple(operator.ge, other)

    def __le__(self, other):
        return self.add_constraint_simple(operator.le, other)

    def __eq__(self, other):
        return self.add_constraint_simple(operator.eq, other)

    def get_value(self):
        return DecisionMatrix(np.vectorize(pulp.value)(self.values), self.index, self.columns, self.model)


if __name__ == '__main__':
    prob = LpProblem()
    lineup = DecisionMatrix.lp_variable('lineup', range(20), range(1, 9), column_type='gw', model=prob)
    bench = DecisionMatrix.lp_variable('bench', range(20), range(9), column_type='gw', model=prob)
    prob += 1
    squad = bench.lag(1) + lineup
    prob += squad <= 1
    print(squad)
    prob.solve()
