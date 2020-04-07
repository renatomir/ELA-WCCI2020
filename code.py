import multiprocessing
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
from sklearn.utils import check_random_state
from scipy.spatial import distance
import math 
import sys
import operator
import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

random.seed()
neighbors_finding = 5			#k = 5 (number of closest neighbors evaluated in the linear regression)

trainingSetName = sys.argv[1]		#training data set
testSetName = sys.argv[2]		#test data set
dimension = int(sys.argv[3])		#number of features
depth_GP = int(sys.argv[4])		#GP depth
generate_graph = int(sys.argv[5])	#0: do not generate graph / 1: generate graph
if generate_graph == 1:
	featuresFileName = sys.argv[6]	#features file name
	num_Interval = int(sys.argv[7])
else:
	featuresFileName = "nofile"
	num_Interval = 10

np.set_printoptions(suppress=True)
# Training samples
training = genfromtxt(trainingSetName, delimiter=',')
X_train = training[:,:dimension]
y_train = np.ravel(training[:,dimension:])

# Testing samples
test = genfromtxt(testSetName, delimiter=',')
X_test = test[:,:dimension]
y_test = np.ravel(test[:,dimension:])

total_train_points= len(y_train)
total_test_points= len(y_test)

######################################################################################
#Calculates the Euclidean distance between the test point (one at a time) and training points

def calculatesDistanceELA(X_test, p, FeaturesSelecionados):
	aux_dist = np.zeros(shape=(total_train_points,2))
	#only positions (attributes) used by symbolic regression
	a = []
	for indice in FeaturesSelecionados:
		a.append((X_test[p])[indice])

	for x in range(0, total_train_points):
		b = []
		for indice in FeaturesSelecionados:
			b.append((X_train[x])[indice])

		aux_dist[x][0] = x	#point number
		aux_dist[x][1] = distance.euclidean(a, b)	#distances between points
	#sorts distances
	aux_dist= aux_dist[np.argsort(aux_dist[:, 1])]
	return aux_dist

######################################################################################
#Symbolic Regression

#fitness
def evalSymbReg(individual, points):
	# Transform the expression of the tree into a function that can be called
	func = toolbox.compile(expr=individual)

	r_simbolica = lambda *x: (func(*x))
	y_pred_reg_simb_t = map(r_simbolica, *points) 

	rmse_test = np.sqrt(np.mean((y_pred_reg_simb_t-y_train)**2))
	return rmse_test,

#toolbox
toolbox = base.Toolbox()

#create GP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))			#negative weight because we want to minimize
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)	#individual: tree and fitness

#primitive set
def aqDiv(left, right):
    return (left / (np.sqrt(1 + (right*right))))

pset = gp.PrimitiveSet("MAIN", arity=dimension) 
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(aqDiv, 2)	#division without discontinuity
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

#Number of participants in each tournament
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", gp.cxOnePoint)
#Each leaf has the same depth between 0 and 2
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.register("evaluate", evalSymbReg, points=np.transpose(X_train))

pool = multiprocessing.Pool()
toolbox.register("map", pool.map)

#depth limit = depth_GP, otherwise you stay with one parent
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=depth_GP))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=depth_GP))

def printResults(Results):
	for item in Results:
		print item,", ", 
	print
	print np.mean(Results)
	print np.std(Results)
	print

def graphicAreaEmpilhada(testSetName, totalintervals, graphicArea, dimension, min_y_test, max_y_test, interval, featuresFileName, num_Interval):
	featuresNames = []
	objetivo = ""
	arq = open(featuresFileName, 'r')
	texto = arq.readlines()
	for linha in texto :
	    featuresNames.append(linha)
	arq.close()

	objetivo = featuresNames.pop()

	#cria pagina html
	nomeArquivo = 'graphic_'+testSetName+'.html'
	arqEscrever = open(nomeArquivo, 'w')

	graphicAreaEmpilhada = """
	<!doctype html>
	<html lang="pt-BR">

	<head>
	<meta charset="utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<title>graphic</title>
	<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.7.1/Chart.min.js"></script>

	</head>

	<body>

	Features:<br>
	"""

	ListFeatures =""
	for nome in featuresNames:
		ListFeatures += nome
		ListFeatures += "<br>"

	graphicAreaEmpilhada += ListFeatures
	graphicAreaEmpilhada += "<br>Goal:<br>"
	graphicAreaEmpilhada += objetivo
	graphicAreaEmpilhada += "<br><br>"

	graphicAreaEmpilhada += "Total test samples evaluated in each interval: "
	for valor in totalintervals:
		graphicAreaEmpilhada += str(valor)
		graphicAreaEmpilhada += " "
	graphicAreaEmpilhada += "<br><br>"

	graphicAreaEmpilhada += """
	<section><canvas id="graphicareas" width="400" height="200"></canvas></section>
	<br><br>

	<script>
	//Time for transition animation
	Chart.defaults.global.animation.duration = 0;
	///////////////////////////////////////////////////////////////////////////////////////////////
	function getRandomColor() {
		    var letters = '0123456789ABCDEF'.split('');
		    var color = '#';
		    for (var i = 0; i < 6; i++ ) {
			color += letters[Math.floor(Math.random() * 16)];
		    }
		    return color;
			}

	var ctx = document.getElementById("graphicareas"); 
	var graphicareas = new Chart(ctx, {
	  type: 'line',
	  data: {
	"""

	#interval vs feature
	for i in range(0,num_Interval):
		if totalintervals[i] != 0:
			graphicArea[i] = graphicArea[i]/totalintervals[i]
		else:	
			graphicArea[i] = 0

	PorcentagensAcumulado = np.zeros((num_Interval, dimension))
	for x in range(0, num_Interval):
		for y in range(0, dimension):
			if y==0:
				PorcentagensAcumulado[x][y] = graphicArea[x][y]
			else:
				PorcentagensAcumulado[x][y] = graphicArea[x][y] + PorcentagensAcumulado[x][y-1]

	saidagraphic = np.transpose(np.around(PorcentagensAcumulado, decimals=2))

	aux = min_y_test
	Labels = "labels: ["
	while aux < max_y_test:
		Labels += str(aux)
		Labels += ", "
		aux += interval
	Labels += "],"

	graphicAreaEmpilhada += Labels
	
	DataSets = """
		datasets: ["""

	i=-1
	for item in saidagraphic:
		i += 1
		DataSets += """{
				"""
		DataSets += "data: ["
		for amostra in item:
			DataSets += str(amostra)
			DataSets += ", "
		DataSets +=  "],"
		DataSets += """
			label: \""""
		DataSets += featuresNames[i].strip("\n")
		DataSets += " \","
		DataSets += """
			backgroundColor: getRandomColor()"""
		DataSets += "},"

	graphicAreaEmpilhada += DataSets

	finalArquivo = """
	    ]
	  },
	  options: {
	    legend: { display: false },
	    title: {
	      display: true,
	      text: 'Importance of features'
	    },
	    scales: {
		yAxes: [{
		  ticks: {
		    //Inserts a percent sign on all values displayed on the x-axisx
		    callback: function(value, index, values) {
			            return value+'%';
			        }
		  }
		}]
	      },

		tooltips: {
		  callbacks: {
		    title: function(tooltipItems, data) {
		      return '"""

	finalArquivo += objetivo.strip("\n")
	finalArquivo += """'+tooltipItems[0].xLabel;        
		    },
		    //Inserts a percent sign
		   label: function(tooltipItems, data) {
		      return data.datasets[tooltipItems.datasetIndex].label+': '+ tooltipItems.yLabel+'%';        
		    }
		  },
		  footerFontStyle: 'normal'
		},
	  }
	});

	</script>

	</body>
	</html>
	"""
	graphicAreaEmpilhada += finalArquivo
	arqEscrever.write(graphicAreaEmpilhada)
	arqEscrever.close()


def main():
	global X_train, dimension, X_test, total_train_points, total_test_points, featuresFileName, num_Interval

	if generate_graph == 0:
		limit = 30
	if generate_graph == 1:
		limit = 1

	results_SymbolicRegression_sizeBestInd = np.zeros(limit)

	results_SymbolicRegression_RMSE_train = np.zeros(limit)
	results_SymbolicRegression_R2_train = np.zeros(limit)
	results_SymbolicRegression_RMSE_test = np.zeros(limit)
	results_SymbolicRegression_R2_test = np.zeros(limit)

	results_RegressaoLinear_RMSE_train = np.zeros(limit)
	results_RegressaoLinear_R2_train = np.zeros(limit)
	results_RegressaoLinear_RMSE_test = np.zeros(limit)
	results_RegressaoLinear_R2_test = np.zeros(limit)

	results_RegressaoLinearL1_RMSE_train = np.zeros(limit)
	results_RegressaoLinearL1_R2_train = np.zeros(limit)
	results_RegressaoLinearL1_RMSE_test = np.zeros(limit)
	results_RegressaoLinearL1_R2_test = np.zeros(limit)

	results_RegressaoLinearL2_RMSE_train = np.zeros(limit)
	results_RegressaoLinearL2_R2_train = np.zeros(limit)
	results_RegressaoLinearL2_RMSE_test = np.zeros(limit)
	results_RegressaoLinearL2_R2_test = np.zeros(limit)

	repetitionCounter = 0
	while repetitionCounter < limit:
		print "Repetition = ", repetitionCounter
		######################################################################################
		print "SYMBOLIC REGRESSION: "

		pop = toolbox.population(n=1000)	#initial population size
		hof = tools.HallOfFame(1)		#keeps the best individual

		stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
		stats_size = tools.Statistics(len)
		mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
		mstats.register("avg", np.mean)
		mstats.register("std", np.std)
		mstats.register("min", np.min)
		mstats.register("max", np.max)

		###GP
		#crossing probability: 0.8
		#probability of mutation: 0.2
		#number of generations: 250
		pop, log = algorithms.eaSimple(pop, toolbox, 0.8, 0.2, 250, stats=mstats, halloffame=hof, verbose=True)

		print "Final equation: "
		best_ind = tools.selBest(pop, 1)[0]
		print best_ind

		#identifies which are the attributes used in the final equation obtained by the symbolic regression
		nodes, edges, labels = gp.graph(best_ind)
		sizeInd = len(nodes)
		FeaturesSelecionados = []
		for item in labels:
			if "ARG" in str(labels[item]):
				FeaturesSelecionados.append(int(str(labels[item]).strip("ARG")))
		FeaturesSelecionados = sorted(set(FeaturesSelecionados))
		for indice in FeaturesSelecionados:
			print indice

		##Y predicted: test and training
		func = toolbox.compile(expr=best_ind)
		r_simbolica = lambda *x: (func(*x))
		y_pred_reg_simb_test = map(r_simbolica, *(np.transpose(X_test)))
		y_pred_reg_simb_train  = map(r_simbolica, *(np.transpose(X_train)))

		#R2 training
		score_gp_train = r2_score(y_pred_reg_simb_train, y_train)
		print('R2 on the training set: %.2f' % score_gp_train)

		#RMSE training
		rmse_train = np.sqrt(np.mean((y_pred_reg_simb_train-y_train)**2))
		print('RMSE on the training set: %.2f' % rmse_train)

		#R2 test
		score_gp = r2_score(y_pred_reg_simb_test, y_test)
		print('R2 on the test set: %.2f' % score_gp)

		#RMSE test
		rmse_test = np.sqrt(np.mean((y_pred_reg_simb_test-y_test)**2))
		print('RMSE on the test set: %.2f' % rmse_test)

		results_SymbolicRegression_RMSE_train[repetitionCounter] = rmse_train
		results_SymbolicRegression_R2_train[repetitionCounter] = score_gp_train
		results_SymbolicRegression_RMSE_test[repetitionCounter] = rmse_test
		results_SymbolicRegression_R2_test[repetitionCounter] = score_gp

		results_SymbolicRegression_sizeBestInd[repetitionCounter] = sizeInd
		print "best individual size = ", sizeInd

		######################################################################################
		print "LINEAR REGRESSION: "

		regr = LinearRegression()
		regr.fit(X_train, y_train)

		r2_train = regr.score(X_train, y_train)		#Returns the coefficient of determination R^2 of the prediction.
		r2_test = regr.score(X_test, y_test)
		print('R2 on the training set: %.2f' % r2_train)
		print('R2 on the test set: %.2f' % r2_test)

		##y predicted training
		y_predicted_rl_train = regr.predict(X_train)

		rmse_test_rl_train = np.sqrt(np.mean((y_predicted_rl_train-y_train)**2))
		print('RMSE on the training set: %.2f' % rmse_test_rl_train)

		##y predicted test
		y_predicted_rl = regr.predict(X_test)

		rmse_test_rl = np.sqrt(np.mean((y_predicted_rl-y_test)**2))
		print('RMSE on the test set: %.2f' % rmse_test_rl)

		# The coefficients
		print('Coefficients: ', regr.coef_)
		print('intercept: ', regr.intercept_)
		print

		results_RegressaoLinear_RMSE_train[repetitionCounter] = rmse_test_rl_train
		results_RegressaoLinear_R2_train[repetitionCounter] = r2_train 
		results_RegressaoLinear_RMSE_test[repetitionCounter] = rmse_test_rl
		results_RegressaoLinear_R2_test[repetitionCounter] = r2_test

		######################################################################################
		print "LINEAR REGRESSION L1: "

		regr = linear_model.Lasso(alpha=1.0)	#alpha: constante que multiplica o termo l1. 
		regr.fit(X_train, y_train)

		r2_train = regr.score(X_train, y_train)		#Returns the coefficient of determination R^2 of the prediction.
		r2_test = regr.score(X_test, y_test)
		print('R2 on the training set: %.2f' % r2_train)
		print('R2 on the test set: %.2f' % r2_test)

		##y predicted training
		y_predicted_rl_train = regr.predict(X_train)

		rmse_test_rl_train = np.sqrt(np.mean((y_predicted_rl_train-y_train)**2))
		print('RMSE on the training set: %.2f' % rmse_test_rl_train)

		##y predicted test
		y_predicted_rl = regr.predict(X_test)

		rmse_test_rl = np.sqrt(np.mean((y_predicted_rl-y_test)**2))
		print('RMSE on the test set: %.2f' % rmse_test_rl)

		# The coefficients
		print('Coefficients: ', regr.coef_)
		print('intercept: ', regr.intercept_)
		print

		results_RegressaoLinearL1_RMSE_train[repetitionCounter] = rmse_test_rl_train
		results_RegressaoLinearL1_R2_train[repetitionCounter] = r2_train 
		results_RegressaoLinearL1_RMSE_test[repetitionCounter] = rmse_test_rl
		results_RegressaoLinearL1_R2_test[repetitionCounter] = r2_test

		######################################################################################
		print "LINEAR REGRESSION L2: "

		regr = linear_model.Ridge(alpha=1.0)
		regr.fit(X_train, y_train)

		r2_train = regr.score(X_train, y_train)		#Returns the coefficient of determination R^2 of the prediction.
		r2_test = regr.score(X_test, y_test)
		print('R2 on the training set: %.2f' % r2_train)
		print('R2 on the test set: %.2f' % r2_test)

		##y predicted training
		y_predicted_rl_train = regr.predict(X_train)

		rmse_test_rl_train = np.sqrt(np.mean((y_predicted_rl_train-y_train)**2))
		print('RMSE on the training set: %.2f' % rmse_test_rl_train)

		##y predicted test
		y_predicted_rl = regr.predict(X_test)

		rmse_test_rl = np.sqrt(np.mean((y_predicted_rl-y_test)**2))
		print('RMSE on the test set: %.2f' % rmse_test_rl)

		# The coefficients
		print('Coefficients: ', regr.coef_)
		print('intercept: ', regr.intercept_)
		print

		results_RegressaoLinearL2_RMSE_train[repetitionCounter] = rmse_test_rl_train
		results_RegressaoLinearL2_R2_train[repetitionCounter] = r2_train 
		results_RegressaoLinearL2_RMSE_test[repetitionCounter] = rmse_test_rl
		results_RegressaoLinearL2_R2_test[repetitionCounter] = r2_test

		######################################################################################
		#To create a graph of stacked areas, discretize the exit interval in num_Interval parts
		min_y_test = np.min(y_test)
		max_y_test = np.max(y_test)
		interval = (max_y_test-min_y_test)/num_Interval
		graphicArea = np.zeros(shape=(num_Interval,dimension))
		totalintervals = np.zeros(num_Interval)

		#Explain the result of the symbolic regression through a LINEAR REGRESSION of the NEAREST POINTS of the training set

		#R2
		r2_score_Real_ELA = np.zeros(total_test_points)
		r2_score_GP_ELA = np.zeros(total_test_points)
		r2_score_Real_GP = np.zeros(total_test_points)

		#Residual_sum_of_squares
		ssr_Real_ELA = np.zeros(total_test_points)
		ssr_GP_ELA = np.zeros(total_test_points)
		ssr_Real_GP = np.zeros(total_test_points)

		#RMSE
		rmse_Real_ELA = np.zeros(total_test_points)
		rmse_GP_ELA = np.zeros(total_test_points)
		rmse_Real_GP = np.zeros(total_test_points)

		for p in range(0, total_test_points):
			newX_train_near = np.zeros(shape=(neighbors_finding,dimension))
			newy_train_near = np.zeros(neighbors_finding)

			#finds the training points closest to the point analyzed at the moment
			distancias_ponto = calculatesDistanceELA(X_test, p, FeaturesSelecionados)
			nearest_points = (distancias_ponto[0:neighbors_finding])[:,0]	#index column only

			for x in range(0, neighbors_finding):
				newX_train_near[x] = X_train[math.trunc(nearest_points[x])]			#X train
				newy_train_near[x] = y_pred_reg_simb_train[math.trunc(nearest_points[x])] 	#Y obtained by symbolic regression in training

			regr = LinearRegression()
			regr.fit(newX_train_near, newy_train_near)

			print ">>> Test point = ",X_test[p]
			print "----- ", "Real value = ", y_test[p]," - Value predicted by the global SYMBOLIC REGRESSION = ",y_pred_reg_simb_test[p]
			
			print "- LINEAR REGRESSION proposta (local explanation):"
			predictedRegressaoLinear = regr.predict(np.array([X_test[p]]))
			print "Predicted Value =", predictedRegressaoLinear
			print('Coefficients: ', regr.coef_)
			print('intercept: ', regr.intercept_)

			#check the importance of each attribute in the regression performed (with absolute values)
			total = np.sum(np.absolute(np.multiply(regr.coef_, X_test[p])))
			importancePercentage = np.divide(np.multiply(np.absolute(np.multiply(regr.coef_, X_test[p])), 100),total)
			print "----- Importance of features: ", importancePercentage

			#For the graph of stacked areas
			if total != 0:
				interval_amostra = math.trunc((y_test[p]-min_y_test)/interval)
				if(y_test[p] == max_y_test):	
					interval_amostra = num_Interval-1

				totalintervals[interval_amostra] += 1	#total test instances in a specific range
				for i in range(0, dimension): 
					graphicArea[interval_amostra][i] += importancePercentage[i]	#sum the importance by features according to the test set

			print "Variation of attributes between neighbors within a maximum range of 10%:"
			#locally interprets the result within a small interval
			localExplanation = np.zeros(shape=(2,dimension)) #minimum and maximum value of each artribute considering the defined interval
			localExplanation[0,:] = X_test[p]
			localExplanation[1,:] = X_test[p]
			
			predictedValues = []
			aux = 0
			for ponto in newX_train_near:
				previsaoPontoVizinho = regr.predict(np.array([ponto]))
				#maximum allowed range to evaluate on the straight (in neighbors) is 10% of the total value in the variation
				if np.absolute(previsaoPontoVizinho-predictedRegressaoLinear) <  interval/2:
					predictedValues.append(previsaoPontoVizinho)
					aux += 1
					for x in range(0, dimension):
						if (ponto[x] < localExplanation[0,x]):
							localExplanation[0,x] = ponto[x]
						if (ponto[x] > localExplanation[1,x]):
							localExplanation[1,x] = ponto[x]

			try:
				print "Considered interval: ",np.min(predictedValues) ," - ",np.max(predictedValues)
			except:
				print "None of the neighbors are within the maximum allowed range (", (predictedRegressaoLinear-interval/2), " - ",  (predictedRegressaoLinear+interval/2), ")"
			for x in range(0, dimension):
				print localExplanation[0,x],"<= Feature ",x," <=",localExplanation[1,x]
			print

			#Y REAL for the training data closest to the analyzed point
			y_real_train = np.zeros(neighbors_finding)
			for x in range(0, neighbors_finding):
				y_real_train[x] = y_train[math.trunc(nearest_points[x])]	#armazena o y real dos points mais near do ponto p avaliado

			#Y obtained by LINEAR REGRESSION with the points closest to the analyzed point
			y_predicted_treino = np.zeros(neighbors_finding)
			y_predicted_treino = regr.predict(newX_train_near)

			#Calculates R2, SSR e RMSE
			#Real value / SYMBOLIC REGRESSION
			r2_score_Real_GP[p] = r2_score(y_real_train, newy_train_near)
			ssr_Real_GP[p] = np.sum((y_real_train - newy_train_near)**2)
			rmse_Real_GP[p] = np.sqrt(np.mean((newy_train_near-y_real_train)**2))

			#Real value / ELA
			r2_score_Real_ELA[p] = r2_score(y_real_train, y_predicted_treino)
			ssr_Real_ELA[p] = np.sum((y_real_train - y_predicted_treino)**2)
			rmse_Real_ELA[p] = np.sqrt(np.mean((y_predicted_treino-y_real_train)**2))

			#SYMBOLIC REGRESSION / ELA
			r2_score_GP_ELA[p] = r2_score(newy_train_near, y_predicted_treino)
			ssr_GP_ELA[p] = np.sum((newy_train_near - y_predicted_treino)**2)
			rmse_GP_ELA[p] = np.sqrt(np.mean((y_predicted_treino-newy_train_near)**2))

		Resultado = np.zeros(18)

		#Real value / SYMBOLIC REGRESSION:
		Resultado[0] = np.mean(r2_score_Real_GP)	#R2
		Resultado[1] = np.std(r2_score_Real_GP)	#deviation R2
		Resultado[2] = np.mean(ssr_Real_GP)	#SSR
		Resultado[3] = np.std(ssr_Real_GP)		#deviation SSR
		Resultado[4] = np.mean(rmse_Real_GP)	#RMSE
		Resultado[5] = np.std(rmse_Real_GP)	#deviation RMSE

		##Real / ELA
		Resultado[6] = np.mean(r2_score_Real_ELA)	#R2
		Resultado[7] = np.std(r2_score_Real_ELA)	#deviation R2
		Resultado[8] = np.mean(ssr_Real_ELA)	#SSR
		Resultado[9] = np.std(ssr_Real_ELA)	#deviation SSR
		Resultado[10] = np.mean(rmse_Real_ELA)	#RMSE
		Resultado[11] = np.std(rmse_Real_ELA)	#deviation RMSE

		##SYMBOLIC REGRESSION / ELA
		Resultado[12] = np.mean(r2_score_GP_ELA)	#R2
		Resultado[13] = np.std(r2_score_GP_ELA)	#deviation R2
		Resultado[14] = np.mean(ssr_GP_ELA)		#SSR
		Resultado[15] = np.std(ssr_GP_ELA)		#deviation SSR
		Resultado[16] = np.mean(rmse_GP_ELA)		#RMSE
		Resultado[17] = np.std(rmse_GP_ELA)		#deviation RMSE

		print "----------------------------------------------------------------------"
		for item in Resultado:
			print item,", ",
		print	
		print "rmse_Real/GP: ", Resultado[4]
		print "rmse_Real/ELA: ", Resultado[10]
		print "rmse_GP/ELA: ", Resultado[16]
		print "----------------------------------------------------------------------"
		
		if generate_graph == 1:
			#creates html page with results
			graphicAreaEmpilhada(testSetName, totalintervals, graphicArea, dimension, min_y_test, max_y_test, interval, featuresFileName, num_Interval)

		repetitionCounter += 1

	printResults(results_SymbolicRegression_sizeBestInd)
	printResults(results_SymbolicRegression_RMSE_train)
	printResults(results_SymbolicRegression_R2_train)
	printResults(results_SymbolicRegression_RMSE_test)
	printResults(results_SymbolicRegression_R2_test)
	printResults(results_RegressaoLinear_RMSE_train)
	printResults(results_RegressaoLinear_R2_train)
	printResults(results_RegressaoLinear_RMSE_test)
	printResults(results_RegressaoLinear_R2_test)
	printResults(results_RegressaoLinearL1_RMSE_train)
	printResults(results_RegressaoLinearL1_R2_train)
	printResults(results_RegressaoLinearL1_RMSE_test)
	printResults(results_RegressaoLinearL1_R2_test)
	printResults(results_RegressaoLinearL2_RMSE_train)
	printResults(results_RegressaoLinearL2_R2_train)
	printResults(results_RegressaoLinearL2_RMSE_test)
	printResults(results_RegressaoLinearL2_R2_test)


if __name__ == "__main__":
    main()
	
