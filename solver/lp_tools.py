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
            self.model = model.model

        if len(self.index) != len(self.values):
            raise Exception(f'DecisionArray index (len={len(self.index)}) and values'
                            f' (len={len(self.values)}) are not the same length')

    @classmethod
    def lp_variable(cls, name, index, column_type=None, column_instance='', cat='Binary', low_bound=None,
                    up_bound=None, model=None):
        if column_type is not None:
            return DecisionSeries(
                data=pulp.LpVariable.dict(f'{name}_{column_type}{column_instance}', index, cat=cat, lowBound=low_bound,
                                          upBound=up_bound), model=model)
        else:
            return DecisionSeries(data=pulp.LpVariable.dict(
                f'{name}', index, cat=cat, lowBound=low_bound, upBound=up_bound), model=model)

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
            return DecisionSeries(data=dict(sorted(data.items())))
        else:
            return DecisionSeries(data={idx: op(self[idx], other) for idx in self.index})

    def __add__(self, other):
        return self.operation(other, operator.add)

    __radd__ = __add__

    def __sub__(self, other):
        return self.operation(other, operator.sub)

    def __mul__(self, other):
        return self.operation(other, operator.mul)

    __rmul__ = __mul__

    def __neg__(self):
        return DecisionSeries(data=-self.values, index=self.index)

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

    def value(self):
        return DecisionSeries(data=[pulp.value(item) for item in self.values], index=self.index)


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
                self.values = np.array([DecisionSeries(data=data[column]).values for column in data]).T
            elif type(data) in (DecisionMatrix, pd.DataFrame):
                self.index, self.columns, self.values = data.index, data.columns, data.values
            elif type(data) in (list, np.ndarray):
                self.index = len(data)

            else:
                raise Exception(f"Data type '{type(data)}' is not valid")
        else:
            self.values, self.index, self.columns = [np.array([])] * 3

        if index is not None:
            self.index = index

        if columns is not None:
            self.columns = columns

        self.model = model

    @classmethod
    def lp_variable(cls, name, index, columns, column_type='', cat='Binary', low_bound=None, up_bound=None, model=None):
        return DecisionMatrix({column: pulp.LpVariable.dict(
            f'{name}_{column_type}{column}', index, low_bound, up_bound, cat) for column in columns}, index, columns)

    def operation(self, other, op):
        pass

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
                return DecisionSeries([row[column_loc] for row in self.values], index=self.index, model=self.model)
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
        return str(pd.DataFrame(data=self.values, index=self.index, columns=self.columns))

    def add_constraint(self, model, relation, other):
        pass


if __name__ == '__main__':
    prob = LpProblem()
    a = DecisionSeries.lp_variable('bench', np.array(range(10)), model=prob)
    b = DecisionSeries.lp_variable('lineup', np.array(range(10)), model=prob)
    c = a + b
    prob += a >= b
    lineup = DecisionMatrix.lp_variable('lineup', range(4, 20), range(1, 9), column_type='gw')

    print(lineup)
    lineup[9] = 10
    print(lineup)
