
def calculate(price, interest_rate, qi):
    m = price
    interest = 0
    passed = 0
    month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    this_month = 0
    for i in range(1, 2 * 365 + 1):
        m += m * interest_rate / 365
        if i - passed == month[this_month]:
            m -= price / qi
            this_month += 1
            this_month %= 12
            passed = i
        print("day " + str(i) + " " + str(m))
    print(m)

calculate(8000, 0.0192, 24)