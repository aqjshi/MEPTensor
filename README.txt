DSCC 395 


Additive and Multiplicative:
    My theory on the implications of Gaussian Theory, you try to find normal distributions of k-components, 
inside a sample space. This is only theoretically valid to apply if the classification problem is a, again, 
classification problem of distributions with a normal distribution. This does not account for skewness of data, 
so to account for that dilema, we increase the amount of components to 'fit' the expected groupings of components
to the actual distribution. So I creae a test to find linear associations of addition based on discrete values. 

Notes: 
    Dont use classifier when doing computation, use Regressor. This avoids the 'static components count' issue

So after using the Regressor we get the output of 
x1,x2,y,y_pred
633,374,1007,941.1407
840,465,1305,1214.8535
367,276,643,609.9940
838,727,1565,1490.3867
107,357,464,443.0168
412,525,937,896.3312
994,504,1498,1392.5699
954,348,1302,1203.3251
616,16,632,602.9312
278,617,895,847.3919
521,232,753,699.0711
136,723,859,831.3783
702,487,1189,1121.6288
351,44,395,369.5300
405,90,495,456.9952
81,980,1061,1037.4225
932,875,1807,1723.4476
98,53,151,141.4135
725,436,1161,1086.0276
4,878,882,866.7858

Mean Squared Error: 0.0011
R^2 Score: 0.9999

Its very impressive, however this is just predicting the linear relationship on addition, its predicting a line, 
and returning a heuristic point on the line. So to truely test the capabilities of ML on determining linear
relationships would be for it to guess y = mx + b, where the output would be a m paired with a b

Notes:
After dealing with the same-point slope, by disabling it, we run tinot another issue, which is inf slope form.
Yielding us 
Mean Squared Error: 1188067749415214110474240.0000
R^2 Score: -0.0862
Terrible, so we disable that one as well. Because we skip the same point and inf slope, we decreased our sample size
from 10000 total combinations to around 9000, a .1 pecent decrease in the amount of unique combinations, we 
compensate by increasing the sample range from 0,10 to 0,20, a small change, but good enough to prove our point that 
its enough to predict a slope. 

Pre-sample enhancement: 1000 iterations
Mean Squared Error: 64.5091
R^2 Score: 0.8820

Post-sample enhancement: 1000 iterations
Mean Squared Error: 101.3186
R^2 Score: 0.8728

Post-sample enhancement: 2000 iterations
Mean Squared Error: 49.7224
R^2 Score: 0.8740

Model is way better because I fixed a typo

Post-sample enhancement: 2000 iterations, 100 hidden layers
Mean Squared Error: 43.9992
R^2 Score: 0.9396

Post-sample enhancement: 4000 iterations, 100 hidden layers
Mean Squared Error: 61.0292
R^2 Score: 0.9014

While, more iterations is supposed to be better, it actually decreases performance of the accuracy of the model, 
especially for linear problems.

Further steps: we will begin by doing differential equations of 2 dimensions, to predict a line of best fit using 
vectors on any line given an interval. for example 3x^3 + x^2 - 5x + 10 on intervals (5,10) inputs would be 
We would have a set of discrete points to train on, 
[[set of points], [5,10]]

and the output would be of length k, where length k would be expected
<3,1,-5,10>

2 approaches, 
1 the matrix approach, which would be similiar to the mnist algorithm, where the line would be represented in 
a 2d array, a 'fingerprint' and we would solve for it. 
2 ODE approach, where we use public sources to compare the performance of our algorithm.


By solving this, we are well on our way of implementing the point made in Attention is all you need. This would build deeper into
from supervised learning to unsupervised learning. This is different than gradient descent problem as it models the 
problem rather than find the min/max- ties in directly with the chirality problem, as we need to model the space, 
rather than navigate and find a exit. 

modulo arithmetic