student = {"국어": 85, "영어": 70, "수학": 63, "과학": 59, "사회": 100}

sum = 0
maximum = 0
subject = ''
for key, value in student.items():
    print("과목 :",key, ",성적 :",student.get(key))
    if value > maximum:
        maximum = value
        subject = key
    sum = sum + value

print("가장 점수가 높은 과목 :",subject, maximum)
print("평균점수 :",sum / len(student))
