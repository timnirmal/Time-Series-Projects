
# Random number set
# Group 5 - 10

# generate number between 1 and 20
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

divided_full = []
divied = []
# Function with start index, end index and step
def range_step(start, end, step):
    while start <= end:
        yield start
        start += step
    return end

"""
# first 4 numbers
for i in range_step(0, 3, 1):
    divied.append(numbers[i])
divided_full.append(divied)

divied2 = []

# last 4 numbers
for i in range_step(4, 8, 1):
    divied2.append(numbers[i])
divided_full.append(divied2)
"""
step_list = [4,8,3,2]

for s in step_list:
    divied = []
    for i in range_step(s, s+3, 1):
        divied.append(numbers[i])
    divided_full.append(divied)



print(divided_full)