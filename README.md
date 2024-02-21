# IPO-EXP
Understanding the connections between the various characteristics and what is done on the day of expiration

Results:
![image_large](https://user-images.githubusercontent.com/24439712/196177194-416aa510-4c86-441a-b633-9cbc3dfb735d.png)



Basic features:(A,B,C,D:necessity rate)

A:
Market Cap,sector,float short,relative volume,shares outstanding
volume,float, price,p/e,price/cash

B:
return on equity,debt/equity, insider ownership,price/free cash,
return on investment,gross margin,insider transaction,peg,operating margin,
institutional ownership,sales grow,inflation rate

C:
eps 5 years, eps indicators...,

D("our features"):
ipoPrice/untilIpoPrice,relational volume(same as float)
worker salary, stock category{0,1}*worker salary
arranging all those features in cols for each day in the last 2 months do next steps:
//every method that used some distribution assumption must be labeled with ASMP
//intentionally i  didn't add any preprocessing
// at the end we can add adaboost kinda algo(weak and strong locs of algos)

1.cov heatmap(may consider spars inverse covariance and minimum cov det)
3. 3 kinds of algos:
supervised
N.N, Dec tree, probabilistic prediction
unsupervised
N.N(boltzman machine)
4.validate each model(cross validate methods)
except from the models create new judging features:
mean difference between prediction and true(bias),number of true and false direction
