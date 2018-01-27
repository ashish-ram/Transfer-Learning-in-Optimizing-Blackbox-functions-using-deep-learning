# Transfer-Learning-in-Optimizing-Blackbox-functions-using-deep-learning
## What is a black-box function?
A system with input and output for which the relationship between in the different variables are not known in a closed mathematical form. There is also no gradient information available. Hence, it is very difficult to optimize such systems. 
## Where are black-box functions are encountered?
In various fields of engineering and science. For example, engineering design problems can be seen as an optimization problem in a very high dimensional space with each point in the space representing a solution. The best design lies at one of global minima. 
## How to optimize a black-box?
The most common method is to build a model of the system by sampling some points and then use this model for optimization. While optimizing, we can sample more points successively near the predicted optima and increase the accuracy. This is called Surrogate Based Analysis and Optimization (SBAO).
 