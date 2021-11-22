print("n을 입력하시오 :", end = " ")
n = int(input())

i = 0
sum = 0
while i <= n:
    if i % 2 == 0:
        sum = sum + i
    i = i + 1

print("1부터 "+str(n)+"까지 짝수의 합", sum)


