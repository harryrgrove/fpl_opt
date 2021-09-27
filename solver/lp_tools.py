import operator
import numpy as np
import pulp
import pandas as pd


class LpProblem:
    """
    Outer class for pulp.LpVariable, same functionality for adding constraints and solving
    """
    def __init__(self, name='NoName', sense=pulp.LpMaximize):
        """
        :param name: Reference name of model
        :param sense: pulp.LpMaximize, pulp.LpMinimize
        """
        self.model = pulp.LpProblem(name, sense)

    def __iadd__(self, other):
        if type(other) == pulp.pulp.LpConstraint:
            self.model += other
        return self

    def __str__(self):
        return str(self.model)

    def solve(self, time_limit=10000, optimizer=pulp.GUROBI, message=True):
        return self.model.solve(optimizer(timeLimit=time_limit, msg=message))


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
            elif type(data) in (int, float):
                if index is not None:
                    self.values = np.repeat(data, len(index))
                else:
                    raise Exception(f'If data type is {type(data)} then index must be specified')
            else:
                raise Exception(f"Data type '{type(data)}' is not valid")
        else:
            self.index, self.values = np.array([]), np.array([])

        if index is not None:
            self.index = np.array(index)

        self.model = model

        if len(self) != len(self.values):
            raise Exception(f'DecisionSeries index (len={len(self)}) and values'
                            f' (len={len(self.values)}) are not the same length')

    @classmethod
    def lp_variable(cls, name, index, column_type=None, column_instance='', low_bound=None, up_bound=None, cat='Binary',
                    model=None):
        """
        Method for generating instance of DecisionSeries from pulp.LpVariable.dict object
        :param name: Name of decision variable
        :param index: Index of DecisionSeries
        :param column_type: Type of column, e.g. GW in FPL. None if column type is not relevant.
        :param column_instance: Nth instance of column, usually placement in DecisionMatrix
        :param low_bound: Lower bound of decision variable contained in self.values
        :param up_bound: Upper bound of decision variable contained in self.values
        :param cat: Category of decision variable contained in self.values
        :param model: LpProblem object of which variable is a part
        :return: DecisionSeries containing pulp.LpVariable objects
        """
        if column_type is not None:
            return DecisionSeries(
                pulp.LpVariable.dict(f'{name}_{column_type}{column_instance}', index, low_bound, up_bound, cat),
                model=model)
        else:
            return DecisionSeries(pulp.LpVariable.dict(f'{name}', index, low_bound, up_bound, cat), model=model)

    def operation(self, other, op, drop=False):
        """
        :param other: array-like or float-like
        :param op: operator from operator library
        :param drop: If True, only columns common to self and other are kept
        :return: operation(self, other) through index where non-common indices are treated as 0
        Operands of different indexes are supported, unmatched values are 0
        If other has no index, must be the same length as self
        """
        # Filter between scalar and element-wise operations
        if hasattr(other, '__getitem__'):
            # Operation is element-wise
            if type(other) in (list, range, np.ndarray):
                if len(self) != len(other):
                    raise Exception('Operands have different lengths')
                data = {self.index[idx]: op(self.values[idx], other[idx]) for idx in range(len(self))}
            else:
                if type(other) in (DecisionSeries, pd.Series):
                    self_index, other_index = self.index, other.index
                elif type(other) == dict:
                    self_index, other_index = self.index, other.keys()
                else:
                    raise TypeError(f"unsupported operand type(s) for {str(op).split(' ')[-1][:-1]}: "
                                    f"'DecisionSeries' and '{type(other)}'")

                data = {idx: op(self[idx], other[idx]) for idx in set(self_index).intersection(set(other_index))}
                if not drop:
                    data |= {idx: op(self[idx], 0) for idx in set(self_index) - set(other_index)} | {
                           idx: op(0, other[idx]) for idx in set(other_index) - set(self_index)}
            return DecisionSeries(dict(sorted(data.items())), model=self.model)
        else:
            return DecisionSeries({idx: op(self[idx], other) for idx in self.index}, model=self.model)

    def __add__(self, other, drop=False):
        return self.operation(other, operator.add, drop)

    __radd__ = __add__

    def __sub__(self, other, drop=True):
        return self.operation(other, operator.sub, drop)

    def __mul__(self, other, drop=False):
        # Element-wise, not matrix multiplication
        return self.operation(other, operator.mul, drop)

    __rmul__ = __mul__

    def __neg__(self):
        """
        :return: Additive inverse of DecisionSeries Vales
        """
        return DecisionSeries(-self.values, self.index, self.model)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        if hasattr(item, '__len__'):
            if len(item) == len(self):
                return DecisionSeries(data=self.values[[index for index, i in enumerate(item) if i == 1]],
                                      index=self.index[[index for index, i in enumerate(item) if i == 1]],
                                      model=self.model)
            else:
                raise ValueError('Invalid index filter')
        else:
            try:
                return self.values[(list(self.index).index(item))]
            except ValueError:
                raise ValueError(f'{item} not in index')

    def __setitem__(self, key, value):
        if key in self.index:
            self.values[(list(self.index).index(key))] = value
        else:
            self.index = np.append(self.index, key)
            self.values = np.append(self.values, value)

    def drop(self, labels, inplace=False):
        if not hasattr(labels, '__iter__'):
            # make all labels iterable
            iter_labels = [labels]
        else:
            iter_labels = labels

        values, index = self.values, self.index
        for label in iter_labels:
            loc = list(index).index(label)
            index, values = np.delete(index, loc), np.delete(values, loc)
        if not inplace:
            return DecisionSeries(values, index, self.model)
        else:
            self.index, self.values = index, values

    def __iter__(self):
        for value in self.values:
            yield value

    def __str__(self):
        """
        :return: string representation of self, same style as pd.Series
        """
        return str(pd.Series(data=[str(value) for value in self.values], index=self.index, dtype='object'))

    def sum(self):
        """
        :return: sum of all elements in self.values
        """
        return pulp.lpSum(self.values)

    def lag(self, value):
        return DecisionSeries(self.values, self.index + value, self.model)

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
        """
        Same functionality as self.add_constraint, with model=self.model is not NoneType
        """
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
        """
        :return: DecisionSeries containing pulp.value(element) for each element in self.values
        """
        return DecisionSeries([pulp.value(item) for item in self.values], self.index, self.model)


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
    def lp_variable(cls, name, index, columns, column_type='', low_bound=None, up_bound=None, cat='Binary', model=None):
        """
        Method for generating instance of DecisionMatrix from pulp.LpVariable.dicts object
        :param name: Name of decision variable
        :param index: Index of DecisionMatrix
        :param columns: Ordered column names of DecisionMatrix
        :param column_type: Type of column, e.g. GW in FPL. None if column type is not relevant.
        :param low_bound: Lower bound of decision variable contained in self.values
        :param up_bound: Upper bound of decision variable contained in self.values
        :param cat: Category of decision variable contained in self.values
        :param model: LpProblem object of which variable is a part
        :return: DecisionMatrix containing pulp.LpVariable objects
        """
        return DecisionMatrix({column: pulp.LpVariable.dict(f'{name}_{column_type}{column}', index, low_bound, up_bound,
                                                            cat) for column in columns}, index, columns, model)

    def operation(self, other, op, drop=False):
        """
        :param other: matrix-like, array-like or float-like
        :param op: operator from operator library
        :param drop: If true, keep non-common columns in output (operation with zero)
        :return: operation(self, other)
        """
        if type(other) in (DecisionMatrix, pd.DataFrame):
            data = {column: op(DecisionSeries(self[column]), DecisionSeries(other[column]))
                    for column in set(self.columns).intersection(other.columns)}
            if not drop:
                data |= {column: op(DecisionSeries(self[column]), 0) for column in set(
                    self.columns) - set(other.columns)} | {column: op(0, DecisionSeries(other[column]))
                                                           for column in set(other.columns) - set(self.columns)}
        else:
            data = {column: op(DecisionSeries(self[column]), other) for column in self.columns}
        return DecisionMatrix(dict(sorted(data.items())), model=self.model)

    def __add__(self, other, drop=True):
        return self.operation(other, operator.add, drop)

    __radd__ = __add__

    def __sub__(self, other, drop=False):
        return self.operation(other, operator.sub, drop)

    def __mul__(self, other, drop=False):
        return self.operation(other, operator.mul, drop)

    __rmul__ = __mul__

    def __neg__(self):
        return DecisionMatrix(-self.values, self.index, self.columns, self.model)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        if type(item) == tuple:
            index, col = item
            return self[col][index]
        elif hasattr(item, '__len__'):
            if len(item) == len(self):
                return DecisionMatrix({col_name: DecisionSeries(col, self.index, self.model)[item] for col, col_name in
                                       zip(self.values.T, self.columns)},
                                      index=self.index[[index for index, i in enumerate(item) if i == 1]],
                                      model=self.model)
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
        elif not hasattr(value, '__getitem__'):
            value = np.repeat(value, len(self))
        elif all(column > key for column in self.columns):
            self.columns = np.insert(self.columns, 0, key, axis=0)
            self.values = np.concatenate([np.array(value)[:, None], self.values], axis=1)
        else:
            self.columns = np.append(self.columns, key)
            self.values = np.append(self.values, np.array(value)[:, None], axis=1)

    def drop(self, labels=None, axis=0, inplace=False):
        """
        :param labels: single label or list-like (labels to be removed)
        :param axis: Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
        :param inplace: If False, return a copy. Otherwise, do operation
        :return: DecisionMatrix with columns in labels parameter removed
        """
        if not hasattr(labels, '__iter__'):
            # make all labels iterable
            iter_labels = [labels]
        else:
            iter_labels = labels

        if axis in (0, 'index'):
            axis_labels = self.index
        elif axis in (1, 'columns'):
            axis_labels = self.columns
        else:
            raise Exception('Invalid axis name')

        values = self.values
        for label in iter_labels:
            loc = list(axis_labels).index(label)
            axis_labels, values = np.delete(axis_labels, loc), np.delete(values, loc, axis)
        if axis in (0, 'index'):
            if not inplace:
                return DecisionMatrix(values, axis_labels, self.columns, self.model)
            else:
                self.index, self.values = axis_labels, values
        elif axis in (1, 'columns'):
            if not inplace:
                return DecisionMatrix(values, self.index, axis_labels, self.model)
            else:
                self.columns, self.values = axis_labels, values

    def __iter__(self):
        for value in self.columns:
            yield value

    def __str__(self):
        return str(pd.DataFrame(data=self.values.astype(str), index=self.index, columns=self.columns))

    def sum(self):
        return DecisionSeries({column: self[column].sum() for column in self.columns}, model=self.model)

    def transpose(self):
        return DecisionMatrix(self.values.T, self.columns, self.index, self.model)

    @property
    def T(self):
        return self.transpose()

    def lag(self, value, axis=1):
        if axis in (1, 'columns'):
            return DecisionMatrix(self.values, self.index, self.columns + value, self.model)
        elif axis in (0, 'index'):
            return DecisionMatrix(self.values, self.index, self.columns + value, self.model)
        else:
            raise Exception('Invalid axis name')

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
    args = {'index': range(20), 'columns': range(1, 9), 'column_type': 'gw', 'model': 'prob'}
    lineup = DecisionMatrix.lp_variable('lineup', **args)
    print(lineup[[False, True] * 10])
    bench = DecisionMatrix.lp_variable('bench', range(20), range(9), column_type='gw', model=prob)
    prob += 1
    squad = bench.lag(1) + lineup
    prob += squad <= 1
