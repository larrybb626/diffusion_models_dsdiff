"""给定两个字符串形式的非负整数 num1 和num2 ，计算它们的和并同样以字符串形式返回。
你不能使用任何內建的用于处理大整数的库（比如 BigInteger）， 也不能使用任何方式将输入的字符串整一个转换为整数形式。
注：可以使用int将每一位进行转换，例如"23"，可以将"2"转成2，"3"转成3去进行运算"""

n1 = '9'
n2 = '1'
def add(num1:str, num2:str) -> str:
    res = []
    i = len(num1) - 1
    j = len(num2) - 1
    c = 0
    while i >=0 or j>=0:
        n1 = int(num1[i]) if i >=0 else 0
        print(n1)
        n2 = int(num2[j]) if j >=0 else 0
        print(n2)
        tmp = n1 + n2 + c
        print(tmp)
        c = tmp // 10
        res.append(str(tmp % 10))
        i,j = i-1, j-1
    res.reverse()
    return "".join(res)
result = add(n1, n2)
print(result)