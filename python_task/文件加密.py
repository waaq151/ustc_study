
# def lock():
#     inName = input("输入文件名：")
#     outName = input("输出文件名：")
#     outfile = open('./python作业/%s'% outName,'wb')
#     with open('./python作业/%s' % inName, 'rb') as infile:
#         in_b = infile.read(1)
#         while in_b:  
#             outfile.write(in_b+b'5')
#             in_b = infile.read(1)       
#     outfile.close()



def lock():
    inName = input("输入文件名：")
    outName = input("输出文件名：")
    outfile = open('./python作业/%s'% outName,'wb')
    with open('./python作业/%s' % inName, 'rb') as infile:
        in_b = infile.read(1)
        while in_b:  
            in_b = int.from_bytes(in_b,byteorder='big')
            out_b = (in_b + 5)%128
            outfile.write(bytes(out_b))
            in_b = infile.read(1)       
    outfile.close()

lock()
