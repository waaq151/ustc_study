#作业2
class MultList():
    def __init__(self):
        self.data = []
        
    def creatlist(self):
        row = eval(input("Enter the number of rows in the list:"))
        for i in range(row):
            row = list(map(float,input("Enter a row:").split()))
            self.data.append(row)
                
    def locateLargest(self):
        data= self.data
        maxi=0
        maxj=0
        for i in range(len(data)):
            for j in range(len(data[i])):
                if data[maxi][maxj]<data[i][j]:
                    maxi=i
                    maxj=j
        print("The location of largest element is at ({},{})".format(maxi,maxj))
    
    def largestson()
        
        
list1 = MultList()
list1.creatlist()
list1.locateLargest()
