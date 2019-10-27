import tkinter as tk
import os
from tkinter import filedialog
from tkinter import messagebox
from tkinter import scrolledtext
class FileWindow:
    def __init__(self):
        window = tk.Tk()

        self.filename = tk.StringVar()
        self.result =tk.StringVar()

        window.title("Occurrence of Letters")
        window.geometry('600x600')

        self.label_filename = tk.Label(window,text='Enter a Filename')
        self.entry_filename = tk.Entry(window,show=None,textvariable=self.filename)
        self.button_Browse = tk.Button(window,text='Browse',command=self.hitBrowse)
        self.button_Result = tk.Button(window,text='show result',command=self.hitResult)
        self.text_Result = scrolledtext.ScrolledText(window,font=("隶书", 14),width=50,height=20)

        self.label_filename.place(x=50,y=570)
        self.entry_filename.place(x=150,y=570)
        self.button_Browse.place(x=350,y=570)
        self.button_Result.place(x=450,y=570)
        self.text_Result.place(x=50,y=50)

        window.mainloop()
    
    def hitBrowse(self):
        default_dir = r"C:\Users\lenovo\Desktop"  # 设置默认打开目录
        name = tk.filedialog.askopenfilename(title=u"选择文件",
                                     initialdir=(os.path.expanduser(default_dir)))
        self.filename.set(name)
    
    def hitResult(self):
        # 打开文件
        if os.path.exists(self.filename.get())==False:
            tk.messagebox.showinfo(title='Warning',message='The File not Exist')
            return
        with open(self.filename.get(), 'r') as infile:
            inWord = infile.read()   
            cntdict=dict()
            for c in inWord:
                if c in cntdict:
                    cntdict[c]=cntdict[c]+1
                else:
                    cntdict[c]=1
        # 输出
        pairs = list(cntdict.items())
        #items = [[x,y] for (y,x) in pairs]
        pairs.sort()
        self.result.set('')
        self.text_Result.delete('1.0', 'end')
        for i in pairs:
            if i[0].isalpha():
                s = str()
                s="{word} appears {count} times".format(word=i[0],count=i[1])
                self.text_Result.insert('end',s)
                self.text_Result.insert('end','\n')
        
    
    
FileWindow()
