import tkinter

class ProcessButtonEvent:
    def __init__(self):
        window = tkinter.Tk()
        label = tkinter.Label(window,text="Welcome  to Python")
        btOK = tkinter.Button(window,text="OK",fg="red",command=self.processOK)
        btCancel=tkinter.Button(window,text="Cancel",bg="yellow",command=self.processCancel)


        label.pack()
        btOK.pack()
        btCancel.pack()

        window.mainloop()

    def processOK(self):
        print("OK")

    def processCancel(self):
        exit()

ProcessButtonEvent().processOK()
   





