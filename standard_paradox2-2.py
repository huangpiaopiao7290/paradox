"""
=====================
@author:曾丹
@time:2024/1/16 15:00
=====================
"""
import random
import numpy as np
import itertools
import csv


def get_paradox_matrix(rows, cols):
    """
    初始化矩阵
    :param rows: 行数
    :param cols: 列数
    :return:
    """
    matrix = []
    for row_i in range(rows):
        row = []
        for col_j in range(cols):
            row.append(random.choice([-1, 1]) * (row_i + 1))
        matrix.append(row)
    return matrix


def get_paradox_matrix_by_given(file):
    """
    从文件中读取矩阵
    :param file: data.csv
    :return:
    """
    matrix = []
    try:
        # 打开文件
        with open(file, 'r') as f:
            csv_reader = csv.reader(f)
            # 遍历
            for line in csv_reader:
                # 剔除空行
                if len(line) == 0:
                    print("文件中存在空行")
                    continue
                integer_list = [int(item) for item in line]
                matrix.append(integer_list)
        f.close()
    except FileNotFoundError:
        print("文件不存在")
    except IOError:
        print("文件读取异常")
    finally:
        # 获取矩阵的维数 上面已经对空行进行处理不用再重复判断col
        row = len(matrix)
        col = 0
        if row == 0:
            quit()
        else:
            col = len(matrix[0])

    return matrix, (row, col)


def get_sub_matrix_and_signTable(mtr, num=None):
    """
    随机挑选子句构成新的子句集
    :param mtr: 原始矩阵
    :param num: 挑选的子句的个数 default = row(mtr) + 1
    :return:
    """
    # 获取mtr的维数
    m, n = len(mtr), len(mtr[0])
    if num is None:
        num = m + 1
    cols = get_unique_random_numbers(0, n, num)
    sub_matrix = [[mtr[i][j] for j in cols] for i in range(m)]
    sub_dim = (len(sub_matrix), len(sub_matrix[0]))
    # 初始化标记表
    sign_table = np.zeros(sub_dim)
    return sub_matrix, sign_table


def sub_matrix_combination(mtr, where_to_save, num=None):
    """
    子句集组合   回溯 + 剪枝
    :param mtr: 原始矩阵
    :param where_to_save: 所有子句集组合的位置
    :param num: 挑选的子句的个数 default = row(mtr) + 1

    :return:
    """
    # 获取mtr的维数
    row, col = len(mtr), len(mtr[0])
    if num is None:
        num = row + 1
    # 列序号结果集
    result = []
    backtracking(col, num, 0, [], result)
    # 将所有组合情况写入文件 sub_matrix.csv
    try:
        with open(where_to_save, 'w', encoding='utf-8') as f:
            for item in result:
                current_line = [[mtr[i][j] for j in item] for i in range(row)]
                # 按行存储
                # current_line = sum(current_line, [])
                f.write(str(current_line))
                f.write('\n')
        f.close()
    except IOError:
        print("写入所有子句集组合时出了点小问题")
    except IndexError:
        print("矩阵索引越界")


def backtracking(n, k, startIndex, path, combination):
    # 终止条件
    if len(path) == k:
        combination.append(path[:])
        return
    # 每次从startIndex开始遍历，然后用path保存取到的节点i
    for i in range(startIndex, n - (k - len(path)) + 1):
        path.append(i)
        backtracking(n, k, i + 1, path, combination)
        # 回溯
        path.pop()


def is_sign_complementary_path(paths, rows_ind, flag_table):
    """
    判断该路径是否包含互补对
    存在互补对:
        不做任何操作
    不存在互补对:
        这条路径的每个元素a在sign_table中对应位置的元素b自增
    返回
    :param paths: 路径组合
    :param rows_ind: 坐标列表
    :param flag_table: 标记表
    :return: none
    """
    # 遍历paths，判断每一条路径
    for path, row_ind in zip(paths, rows_ind):
        # 去重
        path = set(path)
        # 判断存在互斥对
        for combination in itertools.combinations(path, 2):
            if combination[0] + combination[1] == 0:
                # print("存在任意两个元素相加等于0")
                break
        else:
            # print("不存在任意两个元素相加等于0")
            # todo: 对这条路径的所有节点的标记加一
            # 获取path中每个节点在原矩阵B的位置
            for col, row in enumerate(row_ind):
                flag_table[row][col] += 1

    return flag_table.tolist()


def update_sub_matrix(sub_mtr_, sign_table):
    """
    根据标记表更新子句集
    :param sub_mtr_: 子句集
    :param sign_table: 当前标记表
    :return: sub_mtr and flag(==-2 代表标记表全零，输出的sub_matrix未修改，程序终止   >= 0 表明标记表存在最大值， 输出的sub_matrix已经剔除最大值 <=> 替换成-1，准备进入下一轮标记)
    """
    # 如果标记表全零 直接输出
    if np.sum(sign_table) == 0:
        return sub_mtr_, -2
    else:
        # 在判断标记值时 应该是判断本轮的标记值 而传入的标记表sign_table是每轮累加的标记值
        # real_sign_table = sign_table
        max_value = np.max(sign_table)
        (row, col) = np.where(sign_table == max_value)
        # 判断最大值的个数
        if len(row) == 1:
            sub_mtr_[int(row)][int(col)] = 0
        else:
            # 多个最大值
            for i, j in zip(row, col):
                sub_mtr_[int(i)][int(j)] = 0
        return sub_mtr_, max_value


def find_map_and_mtr(mtr, combs):
    """
    找到路径中的节点在原矩阵中的坐标
    :param mtr: 子句集矩阵
    :param combs: 路径组合
    :return: list 行坐标列表，每个行坐标在list的索引即是自己的列坐标
    """
    # mrt.T
    mtt = transpose(mtr)
    # 行坐标列表
    row_index = []
    # 行数 列数
    m, n = len(combs), len(mtt)
    for i in range(m):
        for col in range(n):
            row_index.append(mtt[col].index(combs[i][col]))
    # 切片
    sub_row_index = [row_index[_: _ + n] for _ in range(0, len(row_index), n)]

    return sub_row_index


def all_combs_traverse(mtr, value_to_exclude=None):
    """
    所有路径组合
    :param mtr: 子句集
    :param value_to_exclude: 该剔除的值
    :return:
    """
    mtr_t = np.array(mtr).T
    if value_to_exclude is None:
        value_to_exclude = 0

    all_combinations = list(itertools.product(*mtr_t))
    # 打印所有组合
    # for combination in all_combinations:
    #     print(combination)
    # 使用列表推导式过滤包含特定值的组合
    filtered_combinations = [combination for combination in all_combinations if
                             not any(value == value_to_exclude for value in combination)]

    return filtered_combinations


def not_single_element_col(matrix):
    """
    判断每一列非零元素，存在某列非零元素小于2，返回false
    :param matrix: 更新后的子句集
    :return: bool
    """
    # 转置
    cols = transpose(matrix)
    # 判断每个列表项中非零元素的个数
    for entry in cols:
        if len([x for x in entry if x != 0]) < 2:
            return False

    return True


def get_unique_random_numbers(start, end, nums):
    """
    在子句集中获取nums个不重复的随机数 范围[start, end]
    :param start:
    :param end: 闭区间 直接传入矩阵列数需要减一
    :param nums: 随机数个数
    :return:
    """
    end -= 1
    # 创建一个集合，用于存储不重复的随机数
    numbers = set()
    # 生成指定范围内的随机数
    for _ in range(nums):
        number = random.randint(start, end)
        # 如果随机数已经存在于集合中，则重新生成
        while number in numbers:
            number = random.randint(start, end)
        numbers.add(number)
    return list(numbers)


def transpose(matrix):
    """
    矩阵转置
    :param matrix:
    :return:
    """
    # 获取矩阵的行数和列数
    rows, cols = len(matrix), len(matrix[0])

    # 创建一个新的空矩阵，用于存储转置后的结果
    transposed_matrix = [[0 for row in range(rows)] for col in range(cols)]

    # 遍历原矩阵中的每个元素，将其对应的元素放到新矩阵中正确的位置上
    for i in range(rows):
        for j in range(cols):
            transposed_matrix[j][i] = matrix[i][j]

    return transposed_matrix


if __name__ == '__main__':

    # todo 这里改下flag的值就可以切换初始化矩阵的方式 方便测试
    flag = 1
    if flag == 1:
        # 手动输入矩阵
        init_mtr, dimension = get_paradox_matrix_by_given("./resource/data.csv")
    else:
        # 初始化模拟矩阵
        init_mtr = get_paradox_matrix(3, 4)
    # 构造子句集
    sub_mtr, sign_mtr = get_sub_matrix_and_signTable(init_mtr)
    # 将所有情况的子句集放入 sub_matrix.txt
    sub_matrix_combination(init_mtr, "./resource/sub_matrix.txt")
    # 迭代次数
    count = 0

    print("原始矩阵：\n", init_mtr)

    # 这里会打印存在某一列非零元素个数小于2的情况   将更新函数与print()换个位置应该就不会多打印一次了
    while not_single_element_col(sub_mtr):
        count += 1
        print("子句集: \n", sub_mtr)
        sign_mtr = np.zeros((len(sub_mtr), len(sub_mtr[0])))
        print("初始标记表: \n", sign_mtr)

        # test = [[-1, -1, -1, 1], [2, -2, 2, 2], [3, 3, 3, -3]]
        # sign = np.zeros((3, 4))
        # 所有路径
        path_combs = all_combs_traverse(sub_mtr)
        # 路径节点坐标
        rows_ = find_map_and_mtr(sub_mtr, path_combs)
        print("路径节点行坐标：\n", rows_)
        sign_res = is_sign_complementary_path(path_combs, rows_, sign_mtr)

        sub_mtr, flg = update_sub_matrix(sub_mtr, sign_mtr)
        print("遍历后的标记表：\n", sign_res, "\t最大值=", flg)
        if flg == -2:
            print("该子句集不存在矛盾对 \t exit...")
            break
        else:
            print("新的子句集：\t")
            print(sub_mtr)
            print("=============================================")
    print("迭代次数=", count)

