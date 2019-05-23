# Bakery Challenge

## Background
A bakery used to base the price of their produce on an individual item cost. So if a customer ordered 10
cross buns then they would be charged 10x the cost of single bun. The bakery has decided to start
selling their produce prepackaged in bunches and charging the customer on a per pack basis. So if the
shop sold vegemite scroll in packs of 3 and 5 and a customer ordered 8 they would get a pack of 3 and
a pack of 5. The bakery currently sells the following products:

| Item name | Item code | Bunches|
|-----------|-----------|----------|
|Vegemite Scroll| VS5|3 @ 6.99, 5 @ 8.99|
| Croissant | CF | 2 @ 9.95, 5@ 16.95, 8 @ 24.95|
|Bluberry Muffin|MB11|3 @ 5.95, 5 @ 9.95 ,9 @ 16.99|

## Task
Given a customer order you are required to determine the cost and pack breakdown for each product.
To save on shipping space each order should contain the minimal number of packs

## Input:
Each order has a series of lines with each line containing the number of items followed by the product
code. An example input:
10 VS5
14 MB11
13 CF

## Output
A successfully passing test(s) that demonstrates the following output:
10 VS5 $17.98
2 x 5 $8.99
14 MB11 $54.8
1 x 8 $24.95
3 x 2 $9.95
13 CF $25.85
2 x 5 $9.95
1 x 3 $5.95

# Solution
Python 3.5 was used to created this solution.
I have created a Class Bakery with different functions to solve this problem. 
This class needs to be initialized with different item, their code and package along with their prices. Print order also needs to be specified.
The __determineItemBreakup contains the algorithm which sequentially starts from lowest possible order amount until the given order amount and finds the minimum number of 
 packages using the biggest possible package size for each such order amount, storing them while it moves to next order amt.
For larger order amounts it uses the previously calculated minimum number of packages  for smaller order amounts. Simultaneously,
 it also keeps track of the package type being used to calculate the minimum number. In the end it prints it out in the format 
 required using printBakeryReceipt function.
 
 # Instructions to Run
 There are no special instructions required. Just run all in jupyter using Python 3.
 
 # Github Render issue 
 Incase Github does not render Jupyter file use the following link to view :
 https://nbviewer.jupyter.org/github/VarunM24/data-science-portfolio/blob/master/Python/Practice/Bakery/Bakery-Challenge.ipynb
