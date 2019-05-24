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
code. An example input:<br>
10 VS5<br>
14 MB11<br>
13 CF

## Output
A successfully passing test(s) that demonstrates the following output:<br><pre>
10 VS5 $17.98<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2 x 5 $8.99<br>
14 MB11 $54.8<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1 x 8 $24.95<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3 x 2 $9.95<br>
13 CF $25.85<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2 x 5 $9.95<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1 x 3 $5.95<br>
</pre>

# Assumption:
In case a person makes an order which cannot be served using specified packages, it will be considered invalid and displayed with Invalid Order.
# Solution
## Environment
Solution was created in native python implementation in the file Bakery-Challenge-Native.py, so python 3.5 will be required to run it.
Jupyter notebook for this problem is also available.

## Description
I have created a Class Bakery with different functions to solve this problem. 
This class will be be initialized with different item, their code and package along with their prices. Print order will also be used in the initialization. 
In jupyter file to give an order define order in a list with each order as a seperate element. In native python file, the order is taken from command line inputs. Then takeOrder function is called from Bakery class with order list as parameter.<br>
The __determineItemBreakup contains the algorithm which sequentially starts from lowest possible order amount until the given order amount and finds the minimum number of 
 packages using the biggest possible package size for each such order amount, storing them while it moves to next order amt.
 For Order amount X, algorithm will start from order amt = min package possible uptil X. For each such amount appropriate minimum
 number of packages can be found by deducting the valid package size from order amount and looking up the minimum number of package
 stored for previous amounts. 
For larger order amounts it uses the previously calculated minimum number of packages  for smaller order amounts. Simultaneously,
 it also keeps track of the package type being used to calculate the minimum number. In the end it prints it out in the format 
 required using printBakeryReceipt function.
 If there are issues with the input they will be displayed as error messages to the user.
 # Instructions to Run
 Download the file by right clicking and selecting save as from this <a href="https://raw.githubusercontent.com/VarunM24/data-science-portfolio/master/Python/Practice/Bakery/Bakery-Challenge-Native.py" >link </a> 
 To run this file on Windows. 
1. Install python 3.5 from this website : https://www.python.org/downloads/. 
2. Either directly double click the file to run it 
OR
Open Command prompt (windows), go to the folder containing Bakery-Challenge-Native.py and run the command :
`py path-to-Bakery-Challenge-Native.py`

On MAC:
1. Install python 3.5 from this website : https://www.python.org/downloads/. 
2. To run your script from the Terminal window 
   Run the command: `python path-to-Bakery-Challenge-Native.py`
3. Or to run your script from the Finder you have two options:
- Drag it to PythonLauncher
- Select PythonLauncher as the default application to open your script (or any .py script) through the finder Info window and double-click it.

Python 3.5 and Jupyter notebook is also available. Jupyter can be installed from this website:
https://jupyter.readthedocs.io/en/latest/install.html<br>
 
 # Github Render issue 
 Incase Github does not render Jupyter file use the following link to view :
 https://nbviewer.jupyter.org/github/VarunM24/data-science-portfolio/blob/master/Python/Practice/Bakery/Bakery-Challenge.ipynb
