user_input = input().strip().split()
x = dict()

for word in user_input:
    if word not in x: x[word] = [len(x), 0]
    x[word][1] += 1

fp = open('Q1.txt', 'w')
for i in x:
    print(i, x[i][0], x[i][1], file=fp)


'''
output = list()
cnt = dict()

for str1 in user_input:
    if str1 not in output:
        output.append(str1)
        cnt[str1] = 1
    else:
        cnt[str1] += 1
x = 0
for i in output:
    print(i, x, cnt[i])
    x += 1
'''
