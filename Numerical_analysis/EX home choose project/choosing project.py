import random

"""
EX1: ALGO
Members of the group Neoray Hagag: 208090738
Elisaf Sinvani: 318551728
and Yair Davidoff: 314792714
"""


#Are_Details
choosing_subject_algo ={1:"Solving linear equations",
2:"Finding roots",
3:"Approximations",
4:"Integration"}

def lottery(p1, p2, p3, list_of_choosing):
	list_of_choosing[0] = p1 % 4
	list_of_choosing[1] = p2 % 4
	list_of_choosing[2] = p3 % 4
	return list_of_choosing


list_of_choosing = [0, 0, 0]
NH_ID = 208090738
AS_ID = 318551728
YD_ID = 314792714

list_of_choosing=lottery(NH_ID,AS_ID,YD_ID,list_of_choosing)

x=0

while x==0:
	"""
	to prevent the choose 0	
	"""
	x = random.choice(list_of_choosing)

print(f'Are random number of research is {x},'
      f' The Field subject is "{choosing_subject_algo[x]}"')



