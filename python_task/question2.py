#作业2
class MultList():
    def __init__(self):
        self.a = []
        
    def creatlist(self):
        row = eval(input("Enter the number of rows in the list:"))
        for i in range(row):
            row = list(map(float,input("Enter a row:").split()))
            self.a.append(row)
            
        
    def locateLargest(self):
        a= self.a
        maxi=0
        maxj=0
        for i in range(len(a)):
            for j in range(len(a[0])):
                if a[maxi][maxj]<a[i][j]:
                    maxi=i
                    maxj=j
        print("The location of largest element is at ({},{})".format(maxi,maxj))
        
        
list1 = MultList()
list1.creatlist()
list1.locateLargest()