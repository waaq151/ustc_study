instr='asfsalasdfk'
cntdict=dict()
for c in instr:
    if c in cntdict:
        cntdict[c]=cntdict[c]+1
    else:
        cntdict[c]=1
print(cntdict)