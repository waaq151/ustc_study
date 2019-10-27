
def question1():
    J = list(input("Please input J:"))
    S = list(input("please input S:"))
    J.sort()
    S.sort()
    NUM = 0
    j=0
    s=0
    while j<len(J):
        if s>= len(S):
            s%=len(S)
        while s<len(S):
            if J[j]==S[s]:    
                while s<len(S) and J[j]==S[s]:
                    NUM+=1
                    s+=1  
                break
            s+=1               
        j+=1

    print(NUM)

question1()

def comb(index, begin, k, A, C):
    # index表示某个组合中的索引，begin表示从数组A中begin位置开始寻找， 
	# n表示组合中个数，A表示原数组，C表示组合数组

    if index==k:
        for i in range(k):
            print(C[i],end=" ")
        print()
        return
    j = begin
    while j<=len(A)-k+index:
        C[index]=A[j]
        comb(index+1,j+1,k,A,C)
        j+=1

def comb1(index, begin, k, A, C):
    # index表示某个组合中的索引，begin表示从数组A中begin位置开始寻找， 
	# n表示组合中个数，A表示原数组，C表示组合数组
    if index==k:
        for i in range(k):
            print(C[i],end=" ")
        print()
        return
    j = begin
    while  j < len(A):
        C[index]=A[j]
        comb1(index+1,j+1,k,A,C)
        j+=1

def question2():
    n = int(input("Please input n:"))
    k = int(input("please input k:"))  
    A = list(range(1,n+1)) 
    v=0
    C = [v for i in range(k+1)]
    comb(0,0,k,A,C)    
    #print("*********************")         
    #comb1(0,0,k,A,C)

question2()






def adjust(L,left,right):
    base = L[left]
    while left<right:
        while left<right and L[right]>=base:
            right-=1
        L[left]=L[right]
        while left<right and L[left]<base:
            left+=1
        L[right]=L[left]
    L[left] = base
    return left
def mysort(L,left,right):
    if left<right:
        key = adjust(L,left,right)
        mysort(L,left,key-1)
        mysort(L,key+1,right)
#L = [1,3,2,4,3,5,9,8,7,421,2]
#mysort(L,0,len(L)-1)
