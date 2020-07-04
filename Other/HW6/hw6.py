# import the library pulp as p 
import pulp as p 
  
# Create a LP Minimization problem 
Lp_prob = p.LpProblem('Problem', p.LpMaximize)  
  
# Create problem Variables  
V = p.LpVariable("V", lowBound = 0)
pi_rock = p.LpVariable("pi_rock", lowBound = 0)
pi_paper = p.LpVariable("pi_paper", lowBound = 0)
pi_scissors = p.LpVariable("pi_scissors", lowBound = 0)
  
# Objective Function 
Lp_prob += V

#R = [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]
#R = [[0, 2, -1], [-2, 0, 1], [1, -1, 0]]
R = [[0.0, 3.84, -1.0], [-3.84, 0.0, 1.0], [1.0, -1.0, 0.0]]
  
# Constraints:
Lp_prob += pi_rock + pi_paper + pi_scissors == 1
for row in R:
	Lp_prob += row[0]*pi_rock + row[1]*pi_paper + row[2]*pi_scissors >= V
  
# Display the problem 
print(Lp_prob) 
  
status = Lp_prob.solve()   # Solver 
print(p.LpStatus[status])   # The solution status 
  
# Printing the final solution 
print(p.value(pi_rock), p.value(pi_paper), p.value(pi_scissors), p.value(Lp_prob.objective))   
print(p.value(pi_rock)**2 + p.value(pi_paper)**2 + p.value(pi_scissors)**2)