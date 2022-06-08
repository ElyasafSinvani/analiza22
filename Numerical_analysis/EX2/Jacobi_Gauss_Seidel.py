import sys

epsilon = 0.000001
print(epsilon)
matrixA = [[4, 2, 0], [2, 10, 4], [0, 4, 5]]
vectorB = [2, 6, 5]


def check_max_Dominant(matrix):
    z = 1
    index_max = []
    for i in matrix:
        print("Search for the maximum limb with an absolute value in a row:")
        print(list(map(abs, i)))
        z = max(list(map(abs, i)))
        for j in range(len(i)):
            if z == abs(i[j]):
                if j in index_max:
                    return False
                index_max.append(j)
    print("Indexes of the maximum organs in each row:")
    print(index_max)
    p = 0
    temp = []
    for i in matrix:
        temp.extend(i)
        print(f'print temp', temp)
        temp.remove(max(list(map(abs, temp))))
        if i[index_max[p]] <= sum(list(map(abs, temp))):
            return False
        p += 1
        temp.clear()
    return index_max


def NewOrderMatrix(information):
    if information:
        print("\n!! There's a dominant diagonal for the matrix. !!\n")
        p = 0
        newmatrix = []
        newvector = []
        for i in range(len(information)):
            newmatrix.append([])
            newvector.append(0)
        for i in information:
            newmatrix[i].extend(matrixA[p])
            newvector[i] = vectorB[p]
            p += 1
        return newmatrix,newvector
    else:
        print("** There is no dominant diagonal in the matrix. **\n")
        for i in range(len(matrixA)):
            for j in range(i, len(matrixA)):
                if abs(matrixA[j][i]) > abs(matrixA[i][i]):
                    print("Pivot_order<---", matrixA, vectorB)
                    matrixA[i],matrixA[j] = matrixA[j], matrixA[i]
                    vectorB[i], vectorB[j] = vectorB[j], vectorB[i]
                    print("Pivot_order--->", matrixA, vectorB)
        return matrixA, vectorB


def zero_posting(matrixA, vectorB):
    """
    first loop in are sol
    :param matrixA: are matrix
    :param vectorB: are vec
    :return: because we have zero posting in are first loop, the vec_sol he is
    the Matrix in place [i,i]
    """
    i = 0
    for j in matrixA:
        vectorB[i] = j[i]
        i += 1


def create_flags_for_vectorB(flags, vectorB):
    """

    :param flags: list of flags
    :param vectorB: are vec sol
    :return: if the flag will be True that's says that the vec in flags[i] is the
    ariveto the end of sol
    """
    for i in range(len(vectorB)):
        flags.append(False)

def end_progrem(vectorB, f, method):
    print(f'!! The method "{method}" are done after {f} iterations !!\nThe solution is: vector B is: {vectorB}')


def cond_check(vectorB, vec_check, flags, f, method):
    if all(flags):
        end_progrem(vectorB, f, method)
        sys.exit()
    else:
        for i in range(len(flags)):
            if flags[i] != True:
                if abs(vectorB[i]-vec_check[i]) <= epsilon:  # are Convergence solution
                    flags[i] = True

def iterations_Jacoby(matrixA, vectorB, flags):
    """
    the iteration of Jacoby
    :param matrixA: are matrix
    :param vectorB: sol to find
    :param vec_check: vec to check with minus epsilon
    :return: are sol
    """
    method = f'Jacoby'
    f = 0
    while f < 100:
        vec_check = vectorB.copy()  # create anther vec to minus with epsilon
        i = 0
        for j in matrixA:
            temp_sum = 0
            if flags[i] != True:  # just if we are doesn't get to the sol of are vec
                for k in range(len(vectorB)):
                    if i != k:
                        temp_sum += j[k] * vec_check[k]
                    else:
                        temp_sum += j[k]
                vectorB[i] = temp_sum
            i += 1
        cond_check(vectorB, vec_check, flags, f, method)  # check if are sol is get to epsilon
        f += 1
        if f == 100:
            print(f'The program did not convene in 100 iterations')
            sys.exit()


def posting_Jacoby(matrixA, vectorB):
    flags = []
    create_flags_for_vectorB(flags, vectorB)  # create flags for check epsilon
    zero_posting(matrixA, vectorB)  # first iteration
    iterations_Jacoby(matrixA, vectorB, flags)


def div_by_vec(matrixA,vectorB):
    """
    :param matrixA: are matrix after order of dominate or pivot
    :param vectorB: the sol vector
    :return: the switch func take the argument from the vec and switch him with
    his place in the matrix like X1 switch with Matrix[1,1] X2 to Matrix[2,2]
    and than on.
    and she's divide ich one of the object with the a param in the line with the
    x
    """
    i = 0
    for j in matrixA:
        for k in range(len(vectorB)):
            if vectorB[i] != 0:
                if k != i:
                    j[k] = (j[k] / vectorB[i]) * -1
                else:
                    j[k] = j[k] / vectorB[i]
        if i != len(vectorB):
            i += 1
    i = 0
    for i in range(len(vectorB)):
        if vectorB[i] != 0:
            vectorB[i] /= vectorB[i]


def switch(matrixA, vectorB):
    temp = 0
    i = 0
    for j in matrixA:
        temp = j[i]
        j[i] = vectorB[i]
        vectorB[i] = temp
        i += 1


def Jacoby(matrixA, vectorB):
    switch(matrixA, vectorB)
    div_by_vec(matrixA, vectorB)
    posting_Jacoby(matrixA, vectorB)


def print_to_change_mat():
    print(f'this is are Matrix{matrixA}\nand this is are vector B{vectorB}')
    print(f'If you want to change the vector or the matrix please click 1,else input ather letter or number\n. ')


def recall_mat_vec(matrixA, vectorB):
    print(f'First we will change the matrix')
    i = 1
    for j in matrixA:
        for k in range(len(vectorB)):
            temp_int = input(f'your new input for Matrix in place:[{i},{k+1}]')
            j[k] = int(temp_int)

    print(f'Now we will change the matrix')
    for k in range(len(vectorB)):
        temp_int == input(f'your new input for vector B place:[{i}]')
        vectorB[k] = int(temp_int)

    return matrixA, vectorB


def iterations_Gauss_Seidel(matrixA, vectorB, flags):
    """
    the iteration of Gauss_Seidel
    :param matrixA: are matrix
    :param vectorB: sol to find
    :param vec_check: vec to check with minus epsilon
    :return: are sol
    """
    method = f'Gauss_Seidel'
    f = 0
    while f < 100:
        vec_check = vectorB.copy()  # create anther vec to minus with epsilon
        i = 0
        for j in matrixA:
            temp_sum = 0
            if flags[i] != True:  # just if we are doesn't get to the sol of are vec
                for k in range(len(vectorB)):
                    if i != k:
                        temp_sum += j[k] * vectorB[k]
                    else:
                        temp_sum += j[k]
                vectorB[i] = temp_sum
            i += 1
        cond_check(vectorB, vec_check, flags, f, method)  # check if are sol is get to epsilon
        f += 1
        if f == 100:
            print(f'The program did not convene in 100 iterations')
            sys.exit()

def posting_Gauss_Seidel(matrixA, vectorB):
    flags = []
    create_flags_for_vectorB(flags, vectorB)  # create flags for check epsilon
    iterations_Gauss_Seidel(matrixA, vectorB, flags)



def Gauss_Seidel(matrixA, vectorB):
    switch(matrixA, vectorB)
    div_by_vec(matrixA, vectorB)
    posting_Gauss_Seidel(matrixA, vectorB)


def menu(matrixA, vectorB):
    print(f'Please choose the way you want to solve\nFor the Jacobi Method Conference 1\nFor Gauss Seidel Conference 2\n')
    choose = input(f'your move..')

    if int(choose) == 1:
        print(f'You"r choosed in Jacobi Method')
        print_to_change_mat()
        change = input(f'your move..')

        if int(change) == 1:
            matrixA, vectorB = recall_mat_vec(matrixA, vectorB)
        matrixA, vectorB = NewOrderMatrix(check_max_Dominant(matrixA))
        Jacoby(matrixA, vectorB)


    elif int(choose) == 2:
        print(f'You"r choosed in Gauss Seidel Method')
        print(f'You"r choosed in Jacobi Method')
        print_to_change_mat()
        change = input(f'your move..')

        if int(change) == 1:
            recall_mat_vec(matrixA, vectorB)
        matrixA, vectorB = NewOrderMatrix(check_max_Dominant(matrixA))
        Gauss_Seidel(matrixA, vectorB)


menu(matrixA, vectorB)