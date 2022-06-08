def exchange(row, row_replace):
    unit_matrix = temp_elementary = make_unit_matrix()
    temp_elementary[row], temp_elementary[row_replace] = unit_matrix[row_replace], unit_matrix[row]
    return temp_elementary
def make_unit_matrix():
    unit_matrix = []
    for i in range(0, size):
        temp = []

        for j in range(0, size):
            if i == j:
                temp.append(1)
            else:
                temp.append(0)
        unit_matrix.append(temp)
    return unit_matrix

def multiply_two_matrix(matrix1, matrix2):
    result = []
    for r in range(len(matrix1)):
        helper_res = []
        for c in range(len(matrix1)):
            helper_res.append(0)
        result.append(helper_res)
    for i in range(len(matrix1)):
        # iterating by column by B
        for j in range(len(matrix2[0])):
            # iterating by rows of B
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    return result
def make_elementary_matrix(pivot, num1, row, col):
    temp_elementary = make_unit_matrix()
    temp_elementary[row][col] = -1 * (num1 / pivot)
    return temp_elementary

def exchange(row, row_replace):
    unit_matrix = temp_elementary = make_unit_matrix()
    temp_elementary[row], temp_elementary[row_replace] = unit_matrix[row_replace], unit_matrix[row]
    return temp_elementary
def gauss_method(matrix):
    mul = make_unit_matrix()
    for r in range(size):
        maxi = abs(matrix[r][r])
        flag = 0
        for c in range(r, size):
            if abs(matrix[c][r]) > maxi:
                maxi = abs(matrix[c][r])
                flag = 1
                c_max = c
        if flag != 0:
            temp_matrix = exchange(r, c_max)
            mul = multiply_two_matrix(temp_matrix, mul)
            matrix = multiply_two_matrix(temp_matrix, matrix)
    for r in range(size):
        for c in range(r, size):
            if matrix[r][r] == 1 and r != c:
                temp_matrix = make_elementary_matrix(matrix[r][r], matrix[c][r], c, r)
                mul = multiply_two_matrix(temp_matrix, mul)
                matrix = multiply_two_matrix(temp_matrix, matrix)
            else:
                temp_matrix = make_unit_matrix()
                if matrix[r][r] < 0:
                    temp_matrix[r][r] = 1 / matrix[r][r]
                else:
                    temp_matrix[r][r] = 1 / matrix[r][r]
                mul = multiply_two_matrix(temp_matrix, mul)
                matrix = multiply_two_matrix(temp_matrix, matrix)
    for r in range(size-1, -1, -1):
        for c in range(r, -1, -1):
            if r != c:
                temp_matrix = make_elementary_matrix(matrix[r][r], matrix[c][r], c, r)
                mul = multiply_two_matrix(temp_matrix, mul)
                matrix = multiply_two_matrix(temp_matrix, matrix)
    return mul

def Spline_Kobe(arr, x_f1):
    hi = []
    gama = []
    mi = []
    di = []
    new_matrix = []
    final_result1 = []
    s_x = 0
    for i in range(len(arr[0])):
        if i == len(arr[0])-1:
            hi.append(0)
        else:
            hi.append(arr[0][i+1] - arr[0][i])
    for i in range(len(hi)):
        if i == len(hi)-1 or i == 0:
            mi.append(0)
            gama.append(0)
        else:
            gama.append(hi[i]/(hi[i]+hi[i-1]))
            mi.append(1-gama[i])
    for i in range(1, len(arr[0])-1):
        di.append((6/(hi[i-1]+hi[i]))*(((arr[1][i+1]-arr[1][i])/hi[i])-((arr[1][i]-arr[1][i-1])/hi[i-1])))
    for i in range(size):
        helper = []
        for j in range(size):
            if i == j:
                helper.append(2)
            elif j == i+1:
                helper.append(gama[j])
            elif j == i-1:
                helper.append(mi[i+1])
            else:
                helper.append(0)
        new_matrix.append(helper)
    multiply_elementary_matrix = gauss_method(new_matrix)
    for i in range(len(multiply_elementary_matrix)+1):
        final_result1.append(0)
    for r in range(len(multiply_elementary_matrix)):
        for c in range(len(multiply_elementary_matrix)):
            final_result1[r] += multiply_elementary_matrix[r][c] * di[c]

    counter = 0
    for i in range(len(arr[0])):
        if i > 1:
            counter += 1
        if arr[0][i] > x_f1:
            x_1 = arr[0][i - 1]
            x_2 = arr[0][i]
            s_x = (((x_2-x_f1)**3)*final_result1[counter-1]+((x_f1-x_1)**3)*final_result1[counter])/(6*hi[counter])+(((x_2-x_f1)*arr[1][i-1]+(x_f1-x_1)*arr[1][i])/hi[counter]) -(((x_2-x_f1)*final_result1[counter-1]+(x_f1-x_1)*final_result1[counter])*hi[counter])/6
            break
    return s_x

if __name__ == '__main__':
    values_table = tuple([[0, 0.5235987756, 0.7853981634, 1.570796327], [0, 0.5, 0.7072, 1]])
    x_f = 1.047197551

    size = len(values_table[0]) - 2
    print(f'The result in Spline_Kobe method is: {Spline_Kobe(values_table, x_f)}')