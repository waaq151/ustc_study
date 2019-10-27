#include <vector>
#include <iostream>
using namespace std; 
void Comb(int index,int begin,int len,int n,int *A,int *C);
int main()
{
    int A[5]={1,2,3,4,5};
    int len=5,n=3; 
    int *C=new int[n+1];  
    Comb(0,0,len,n,A,C);   
    delete []C; return 0;
} //递归组合
void Comb(int index,int begin,int len,int n,int *A,int *C)
{
 // index表示某个组合中的索引，begin表示从数组A中begin位置开始寻找， 
 // len表示数组A长度，n表示组合中个数，A表示原数组，C表示组合数组  
    if(index==n)  
    {
        for(int i=0;i<n;i++)
            cout<<C[i]<<" "; 
            cout<<endl; 
        return;    
    } 
    for(int j=begin;j<=len-n+index;j++)    
    {   
        C[index]=A[j];         Comb(index+1,j+1,len,n,A,C);    
    }
}
