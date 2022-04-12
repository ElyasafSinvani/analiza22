class derivative():
    def __init__(self, str):
        self.str = str
        self.operator_list = []
        list_polynomials = list(self.str)
        # extract all the plus and minus signs
        for i in list_polynomials:
            if i == "+" or i == "-":
                self.operator_list.append(i)
        # extract the variable
        for x in list_polynomials:
            if x.isalpha():
                self.va_s = x

    # separate the polynomial into monomials
    def separate(self):
        # put all the monomials into a list, called self.list_no_operator
        self.list_no_operator = self.str.split("+")
        for monomial in self.list_no_operator:
            if "-" in monomial:
                term_l = monomial.split("-")
                if "" in term_l:
                    term_l.pop(0)
                place = self.list_no_operator.index(monomial)
                self.list_no_operator = self.list_no_operator[:place] + term_l + self.list_no_operator[place + 1:]
        if self.list_no_operator[0].isdigit():
            return True
        else:
            return False

    # remove the constants, if the constant is not the first monomial of the polynomial, then also remove the operator in front of it
    def remove_constants(self):
        list_no_operator2 = []
        for i in self.list_no_operator:
            list_no_operator2.append(i)
        for i in list_no_operator2:
            if i.isdigit():
                if i in self.list_no_operator:
                    # if the constant is at the beginning of the polynomial and it is larger than 0, only this constant itself should be removed
                    if self.list_no_operator.index(i) == 0 and list(self.str)[0] != "-":
                        self.list_no_operator.remove(i)
                    # if the constant a t the beginning is smaller than 0, then the conastant and the minus sign should both be removed
                    elif self.list_no_operator.index(i) == 0 and list(self.str)[0] == "-":
                        self.list_no_operator.remove(i)
                        self.operator_list.pop(0)
                    # in other case: the constant and the add or minus sign should also be removed
                    else:
                        self.operator_list.pop(self.list_no_operator.index(i) - 1)
                        self.list_no_operator.remove(i)

    # make all the monomials in the form as a*x^b
    def modify_polynomials(self):
        for monomial in self.list_no_operator:
            list_monomial = list(monomial)
            if "*" not in list_monomial:
                place = self.list_no_operator.index(monomial)
                self.list_no_operator[place] = "1*" + monomial
                monomial = "1*" + monomial

            if "^" not in list_monomial:
                place = self.list_no_operator.index(monomial)
                self.list_no_operator[place] = monomial + "^1"

    # get the derivative of each monomial
    def get_each_derivative(self):
        self.term = []
        for monomial in self.list_no_operator:
            term_derivative = str(
                float(monomial.split("*")[0]) * int(monomial.split("^")[1])) + "*" + self.va_s + "^" + str(
                int(monomial.split("^")[1]) - 1)
            self.term.append(term_derivative)

    # put each derivative of monomials and the plus or minus signs together in sequence
    def get_total_derivative(self):
        result = ""
        # if the first monomial of the polynomial is a constant
        if self.separate() and self.str.split("*")[0].isdigit() == False:
            length = len(self.operator_list)
            n = 0
            while n < length:
                result += self.operator_list[n] + self.term[n]
                n += 1
            # extract the plus sign if it appears in the beginning of the polynomial
            if list(result)[0] == "+":
                result = list(result)
                result.pop(0)
                result = "".join(result)
        # if the first character of the polynomial is "-" and the first monomial of the polynomial isn't a constant
        elif list(self.str)[0] == "-" and not self.separate():
            length = len(self.operator_list)
            n = 0
            while n < length:
                result += self.operator_list[n] + self.term[n]
                n += 1
        # if the first character of the polynomial is not "-" and the first monomial of the polynomial isn't a constant
        else:
            self.operator_list.append("")
            length = len(self.operator_list)
            n = 0
            while n < length:
                result += self.term[n] + self.operator_list[n]
                n += 1
        return result


def Pstr_to_Fp(s):
    a = str(s).replace('+', ',+').replace('-', ',-').split(',')

    def F(x):
        result = 0
        for i in a:
            if 'x' not in i:
                result += float(i)
            else:
                result += float(i.split('*x^')[0]) * x ** float(i.split('*x^')[1])
        return result

    return F


def find_roots(f, a, b):
    tol = 10 ** -9
    while b - a > tol:
        m = a + (b - a) / 2
        if f(a) * f(m) > 0:
            a = m
        else:
            b = m
    if int(f((a + b) / 2)) == 0:
        return (a + b) / 2


def roots(Sf, a, b):
    f = Pstr_to_Fp(Sf)
    f_tag = Pstr_to_Fp(Derivative(Sf))
    list_roots = []
    while b > a:
        if f(a) * f(a + 0.1) < 0:
            list_roots += [round(find_roots(f, a, a + 0.1), 9)]
        if f_tag(a) * f_tag(a + 0.1) < 0:
            x = round(find_roots(f_tag, a, a + 0.1), 9)
            if f(x) == 0:
                list_roots += [x]
        a = a + 0.1
    return list_roots


def Derivative(p):
    a = derivative(p)
    a.separate()
    a.remove_constants()
    a.modify_polynomials()
    a.get_each_derivative()
    return a.get_total_derivative()


P = input("Please enter your polynomial :\n")
a = float(input("Please enter the beginning of the range :\n"))
b = float(input("Please enter the end of the range :\n"))
print(roots(P, a, b))
