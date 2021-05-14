import os

def compile_lines(folderPath):
    temp = ""
    output = ""
    for file in os.listdir(folderPath):
        if file.endswith(".txt"):
            f = open(folderPath + "\\" + file, "r")
            temp += f.read()
            f.close()
    for i in range(len(temp)):
        if not ((temp[i-1] == '\n') and (temp[i] == '\n')):
            output += temp[i]
    return output


allBars = compile_lines("C:\\Users\\destr\\Desktop\\bars")

half0 = ""
half1 = ""
i = 1
for line in allBars.splitlines():
    if len(line) < 256:
        if i == 1:
            half0 += line + '\n'
            i = -i
        else:
            half1 += line + '\n'
            i = -i

print(half0)
print(half1)