library(IOHexperimenter)
initiate_parents <- function(dimension, population_size){
  # get first parent population
  population = matrix(data = NA, nrow = population_size, ncol = dimension)
  for(i in 1:(dim(population)[1])){population[i,] = sample(c(0, 1), dimension, TRUE)}
  return(population)
}

eval_population <- function(obj_func, population){
  # This function takes an evaluation function and
  # a population matrix and returns their fitness in a vector
  fitness = obj_func(population)
  return(fitness)
}

tournament <- function(obj_func, population, count=50){
  # get fitness score for each instance:
  fitness_vector = eval_population(obj_func, population)
  # get instance size:
  population_size = dim(population)[1]
  # Get the number of participants
  participants_count =  as.integer(population_size/10)
  # initalize new population matrix
  new_population = matrix(data = NA, nrow = count, ncol = dim(population)[2])
  # run tournaments:
  for(i in 1:count){
    # get sample:
    population_size = dim(population)[1]
    # get best from sample:
    winner_score = max(fitness_vector)#[selected_rows])
    winner = population[which.max(fitness_vector), ]
  }
  return(winner)#new_population
  # runs a tournament and returns a the winners as a matrix
}

mutate <- function(ind, mutation_rate){
  # Inout parameters ind: Individual from the population which will get mutated
  # with the probabilty given as mutation_rate
  # mutation_rate: is the probaility of a bit to be 1
  dim <- length(ind)
  mutations <- seq(0, 0, length.out = dim)
  while (sum(mutations) == 0){
    mutations <- sample(c(0, 1), dim, prob = c(1-mutation_rate, mutation_rate), replace = TRUE)
  }
  return(as.integer( xor(ind, mutations) ))
}

mutate_pop <- function(population, mutation_rate){
  # Mutates each individual using mutate function
  mutated_population = apply(population, 2, function(x) mutate(x, mutation_rate = mutation_rate))
  return(mutated_population)
}

GA <- function(IOHproblem){
  # Establish mutation rate
  # Mutation rate 1/100, 2/100, 10/100
  mutation_rate =  10 / IOHproblem$dimension 
  # Initiate initial parent population:
  best_parents = initiate_parents(IOHproblem$dimension, 1)
  # First selection of best parents
  iterations = 0
  if(iterations == 0){
    cat(paste('\rFunction: ',IOHproblem$function_id))
  }
  budget = 50000
  fitness = c()
  while((IOHproblem$target_hit() == 0) && (iterations < budget)){
    iterations = iterations + 1
    children = best_parents
    parents = mutate_pop(children, mutation_rate = mutation_rate)
    parents = rbind(children, parents, deparse.level = 0)
    best_parents = t(as.matrix(tournament(IOHproblem$obj_func, parents, 1)))
	fitness = append(fitness, eval_population(IOHproblem$obj_func, best_parents))
  }
  filename = sub("DIM",IOHproblem$dimension,"F-23-dim-DIM.csv")
  write.table(fitness, filename, append=T, row.names = F, sep='\t', col.names = F)
  return(list(xopt = best_parents, fopt = IOHproblem$fopt))
}

benchmark_algorithm(GA, functions = 23,repetitions = 1, instances = c(1:5), dimensions = c(16, 25, 36, 49, 64), suite='PBO',
                    algorithm.name = "AML_EA", data.dir = "./AML_EA")
