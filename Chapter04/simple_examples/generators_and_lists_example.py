gen1 = (x**2 for x in range(10))
for i in gen1:
    print(i)


# Pipeline

data_vals = [x for x in range(100)]


def filter_data(data, condition):
    x: object
    for x in data:
        if condition(x):
            yield x


for x in filter_data(data_vals, lambda x: x > 50):
    print(x)


# List Comprehension

data_vals = [x]