import numpy as np


class DecisionTreeRegressor:

    def __init__(self, max_depth=None, min_leaf_samples=None):
        self.max_depth = max_depth
        self.min_leaf_samples = min_leaf_samples
        self._node = {
            'left': None,
            'right': None,
            'feature': None,
            'threshold': None,
            'depth': 0,
            'value': None
        }
        self.tree = None  # словарь в котором будет храниться построенное дерево

    def fit(self, X, y):
        self.tree = {'root': self._node.copy()}  # создаём первый узел в дереве
        self._build_tree(self.tree['root'], X, y)  # запускаем рекурсивную функцию для построения дерева
        return self

    def predict(self, X):
        preds = np.zeros(shape=(X.shape[0],))
        for i, x in enumerate(X):
            preds_for_x = self._get_predict(
                self.tree['root'],
                x
            )  # рекурсивно ищем лист в дереве соответствующий объекту
            preds[i] = preds_for_x
        return preds

    def get_best_split(self, X, y):
        y = y.reshape(-1)
        n, m = X.shape
        best_Q = float("inf")
        best_j = 0
        best_t = 0
        best_left_ids = 0
        best_right_ids = 0
        found_split = False
        y_squared = y * y  # готовим y^2
        
        for j in range(m):  # перебор признаков
            feat = X[:, j]
            order = np.argsort(feat)
            feat_sorted = feat[order]
            y_sorted = y[order] 
            y_squared_sorted = y_squared[order]
            
            # префиксные суммы
            pref_y = np.cumsum(y_sorted)
            pref_y_squared = np.cumsum(y_squared_sorted)
            total_y = pref_y[-1]
            total_y_squared = pref_y_squared[-1]

            # перебор всех сплитов
            for i in range(1, n):
                if feat_sorted[i - 1] == feat_sorted[i]:
                    continue
                left_n = i
                right_n = n - i

                # считаем суммы слева/справа
                left_sum = pref_y[i - 1]
                left_sum_squared = pref_y_squared[i - 1]
                right_sum = total_y - left_sum
                right_sum_squared = total_y_squared - left_sum_squared

                # считаем SSE слева/справа
                left_sse = left_sum_squared - (left_sum * left_sum) / left_n
                right_sse = right_sum_squared - (right_sum * right_sum) / right_n

                # считаем качество сплита и обновляем лучший
                Q = (left_sse + right_sse) / n
                if Q < best_Q:
                    best_Q = Q
                    best_j = j
                    best_t = (feat_sorted[i - 1] + feat_sorted[i]) / 2.0
                    best_left_ids = feat <= best_t
                    best_right_ids = ~best_left_ids
                    found_split = True

        return best_j, best_t, best_left_ids, best_right_ids, found_split

    def _build_tree(self, curr_node, X, y):
        if curr_node['depth'] == self.max_depth:  # выход из рекурсии если построили до максимальной глубины
            curr_node['value'] = y.mean()  # сохраняем предсказания листьев дерева перед выходом из рекурсии
            return

        if len(np.unique(y)) == 1:  # выход из рекурсии значения если "y" одинаковы для все объектов
            curr_node['value'] = y.mean()
            return
        
        if self.min_leaf_samples is not None: # не делаем сплит если один из ребенков после сплита меньше L
            if len(y) < 2 * self.min_leaf_samples:
                curr_node['value'] = y.mean()
                return
        
        j, t, left_ids, right_ids, found_split = self.get_best_split(X, y)  # нахождение лучшего разбиения
        if not found_split:
            curr_node['value'] = y.mean()  # сохраняем предсказания листьев дерева перед выходом из рекурсии
            return
        
        curr_node['feature'] = j  # признак по которому производится разбиение в текущем узле
        curr_node['threshold'] = t  # порог по которому производится разбиение в текущем узле

        left = self._node.copy()  # создаём узел для левого поддерева
        right = self._node.copy()  # создаём узел для правого поддерева

        left['depth'] = curr_node['depth'] + 1  # увеличиваем значение глубины в узлах поддеревьев
        right['depth'] = curr_node['depth'] + 1

        curr_node['left'] = left
        curr_node['right'] = right

        self._build_tree(left, X[left_ids], y[left_ids])  # продолжаем построение дерева
        self._build_tree(right, X[right_ids], y[right_ids])

    def _get_predict(self, node, x):
        # если в узле нет порога, значит это лист, выходим из рекурсии
        if node['threshold'] is None:
            return node['value']
        
        # уходим в правое или левое поддерево в зависимости от порога и признака
        if x[node['feature']] <= node['threshold']:
            return self._get_predict(node['left'], x)
        else:
            return self._get_predict(node['right'], x)


class MyGradientBoostingRegressor:

    def __init__(self, learning_rate, max_depth, n_estimators, min_leaf_samples=None):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.min_leaf_samples = min_leaf_samples
        self.init_value = 0
        self.trees = []

    def fit(self, X, y):
        y = y.reshape(-1)
        self.init_value = np.mean(y)
        pred = np.full(X.shape[0], self.init_value)
        
        for _ in range(self.n_estimators):
            residuals = y - pred
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_leaf_samples=self.min_leaf_samples
            )
            tree.fit(X, residuals)
            pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
        return self

    def predict(self, X):
        preds = np.full(X.shape[0], self.init_value)
        for tree in self.trees:
            preds += self.learning_rate * tree.predict(X)
        return preds


def read_matrix(n, dtype=float):
    matrix = np.array([list(map(dtype, input().split())) for _ in range(n)])
    return matrix

def read_input_matriсes(n, m, k):
    X_train, y_train, X_test = read_matrix(n), read_matrix(n), read_matrix(k)
    return X_train, y_train, X_test

def print_matrix(matrix):
    for row in matrix:
        print(str(row))

def solution():
    n, m, k = map(int, input().split())
    X_train, y_train, X_test = read_input_matriсes(n, m, k)
    y_train = y_train.reshape(-1)

    gb = MyGradientBoostingRegressor(
        learning_rate=0.1,
        max_depth=3,
        n_estimators=300
    )
    gb.fit(X_train, y_train)

    predictions = gb.predict(X_test)
    print_matrix(predictions)

solution()