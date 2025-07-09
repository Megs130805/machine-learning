A = [[1, 1], [2, 2]]
B = [[3, 3], [4, 4]]

if len(A[0]) != len(B):
    print("Error")
else:
    result = []

    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            s = 0
            for k in range(len(B)):
                s += A[i][k] * B[k][j]
            row.append(s)
        result.append(row)

    print("product:", result)
