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

    def MSE(self, y):
        y = y.flatten()
        return np.mean((y - y.mean()) ** 2)

    def calc_Q(self, y_left, y_right):
        n = len(y_left) + len(y_right)
        return (len(y_left) / n) * self.MSE(y_left) + (len(y_right) / n) * self.MSE(y_right)

    def get_best_split(self, X, y):
        m = len(X[0])  # кол-во признаков
        n = len(X)  # кол-во объектов
        best_err = float("inf")
        best_j = 0
        best_t = 0
        best_left_ids = 0
        best_right_ids = 0
        found_split = False

        for idx in range(m):
            feat = X[:, idx]
            order = np.argsort(feat)
            feat_sorted = feat[order]

            thrs = np.unique([(feat_sorted[i] + feat_sorted[i + 1]) / 2 for i in range(n - 1)])
            for t in thrs:
                mask_l = X[:, idx] <= t
                mask_r = ~mask_l
                if self.min_leaf_samples is not None:
                    if mask_l.sum() < self.min_leaf_samples or mask_r.sum() < self.min_leaf_samples:
                        continue # если при разбиении в одном из поддеревьев объектов меньше чем задано
                found_split = True
                err = self.calc_Q(y[mask_l], y[mask_r])
                if err < best_err:
                    best_err = err
                    best_j = idx
                    best_t = t
                    best_left_ids = mask_l
                    best_right_ids = mask_r

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


def read_matrix(n, dtype=float):
    matrix = np.array([list(map(dtype, input().split())) for _ in range(n)])
    return matrix


def read_input_matriсes(n, m, k):
    X_train, y_train, X_test = read_matrix(n), read_matrix(n), read_matrix(k)
    return X_train, y_train, X_test


def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))


def solution():
    n, m, k = map(int, input().split())
    X_train, y_train, X_test = read_input_matriсes(n, m, k)

    clf = DecisionTreeRegressor(max_depth=2)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    print_matrix(preds)


# solution()

# test
# 4 1 3
# 1
# 2
# 4
# 6
# 1
# 3
# 2
# 2
# 1
# 1.5
# 2

