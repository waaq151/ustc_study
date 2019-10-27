def set_zeros(A):
    # 通过两个标志分别记录第一行/列是否有0存在
    tagi=False
    tagj=False
    for i in range(len(A)):
        if A[i][0] == 0:
            tagi = True
    for j in range(len(A[0])):
        if A[0][j] == 0:
            tagj = True
    
    # 从第1行第1列开始遍历，把第一行/列来作为标记，记录该行/列是否需要归零
    for i in range(1,len(A)):
        for j in range(1,len(A[i])):
            if A[i][j] == 0:
                A[i][0] = 0
                A[0][j] = 0
    
    # 根据记录从第1行第1列开始将对应的行/列归零
    for i in range(1,len(A)):
        for j in range(1,len(A[i])):
            if A[i][0]==0 or A[0][j]==0:
                A[i][j]=0
    
    # 最后根据tag确定第一行/列是否需要归零
    if tagi:
        for i in range(len(A)):
            A[i][0] = 0
    if tagj:
        for j in range(len(A[0])):
            A[0][j] = 0
    
    return A
            
def myprint(A):
    for i in range(len(A)):
        print(A[i])

A1 = [[0,1,1,0],
      [1,1,0,1],
      [1,1,1,1]]

set_zeros(A1)
myprint(A1)

# A2 = [[1,1,1,1],
#       [1,1,0,1],
#       [1,0,1,1]]

# set_zeros(A2)
# myprint(A2)







# def set_zeros(A):
#     I = [] # 存储为零的行
#     J = [] # 存储为零的列
#     for i in range(len(A)):
#         for j in range(len(A[i])):
#             if A[i][j] == 0:
#                 I.append(i)
#                 J.append(j)

#     # 
#     for i in I:
#         for j in range(len(A[0])):
#             A[i][j] = 0
    
#     for j in J:
#         for i in range(len(A)):
#                 A[i][j] = 0

#     return A



















