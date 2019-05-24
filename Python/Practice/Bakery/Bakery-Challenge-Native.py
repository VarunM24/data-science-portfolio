
# coding: utf-8

# # Bakery - Orders
# In this problem we have to design an algorithm that will help bakery find the appropriate number and size of prepackaged bunch of items to be used for a specific order amount such that they use the greatest bunch most in order to save shipping money.<br>
# This bakery has 3 menu items and the following bunches available:<br>
# 
# | Item name | Item code | Bunches|
# |-----------|-----------|----------|
# |Vegemite Scroll| VS5|3 @ 6.99, 5 @ 8.99|
# | Croissant | CF | 2 @ 9.95, 5@ 16.95, 8 @ 24.95|
#     |Bluberry Muffin|MB11|3 @ 5.95, 5 @ 9.95 ,9 @ 16.99|
# 
# 
# 
# 
# 
# 

# In[13]:


class Bakery:
    def __init__(self,menu_items, menu_price_dict,printOrder):
        '''Initialize Bakery class with menu items, their item code and package with prices
        Parameters
        ----------
        menu_items: dict
            This is expected to contain item code (str) as key and item name as value
        menu_price_dict: dict
            This is expected to contain item code (str) as key and dictionary of package (int):price(float) as value
        printOrder: list
            Contains item codes (str) in order in which it is expected to be printed
             
        
        '''
#         menu_items = {'Vegemite Scroll':'VS5','Blueberry Muffin':'MB11','Croissant':'CF'}
        assert len(menu_price_dict)==len(menu_items), 'Number of items in menu and menu price do not match.'
        self.__items=menu_items
        self.__items_price_dict = menu_price_dict
        self.__valid_package_dict = {k:sorted(list(v.keys())) for k,v in menu_price_dict.items()}
        self.__printOrder = printOrder
        

    def __determineItemBreakup(self,valid_package_sizes,order_amount,min_package_req_for_diff_amts,pkgs_used):
        ''' Finds the minimum number of biggest packages required for provided order amounts to save space and
        returns the packages found
        Parameters
        ----------
        valid_package_sizes: list
            ascending list of valid package size
        order_amount: int
            number of items ordered for a particular menu item
        min_package_req_for_diff_amts: list
            initialized list of [0] with number of elements = order_amount +1 . Used to store minimum no. of packages 
            required for different order amts
        pkgs_used: list
            initialized list of [0] with number of elements = order_amount +1 . Used to track the type of packages for different
            order amounts which is used to calculate the different types of package used for an order amount
        
        Returns:
        -------
        pkgs_used: list
            list which now contains the package types for different order amounts which is used to 
            calculate the different types of package used for an order amount by __getPackagesUsedAndTotalForAmt method

        '''

        # we will try to find minimum package count for the each of different number of order amounts upto the required ordered amount
        # eg. if ordered amt = 10 then we will find min pkgs req for all amt till 10 starting from the smallest package
        # eg. if smallest package is 3 then from 3,4,5,...10
        # We do this by taking each of these order amts (3,4,5..10) and then we try to find the minimum no. of pkgs required by trying out 
        # each of the given valid sizes. 
        for temp_amt in range(min(valid_package_sizes),order_amount+1):
            #Initially assume maximum (approx) number of smallest packages
            pcks_used_count = temp_amt//min(valid_package_sizes)
            new_pkg =0
            validAmt = False
            # Now we will try to work our way upto the required package amount  and find minimum package numbers required
            # and store that for final output

            for valid_pk_sz in [valid_pk_size for valid_pk_size in valid_package_sizes if valid_pk_size<= temp_amt]:

                # So we assume we have used 1 of the valid pkg (which is why we deducting valid pkg in list)
                # And then we find minimum pkgs stored in list for amt - a valid pk size
                # If minimum package is the initialized value then we 
                # and store it in a list

                # These conditions ensure that the amount order can be served using the available valid package size list
                # Last condition ensure that minimum number of packages are chosen 

                if( ( ((temp_amt - valid_pk_sz >= min (valid_package_sizes))  & (min_package_req_for_diff_amts[temp_amt - valid_pk_sz] >0)) | 
                     (temp_amt - valid_pk_sz ==0) ) & 
                   (min_package_req_for_diff_amts[temp_amt - valid_pk_sz] + 1 <= pcks_used_count)):

                    pcks_used_count = min_package_req_for_diff_amts[temp_amt - valid_pk_sz] + 1
                    # recording the valid package size used
                    new_pkg = valid_pk_sz
                    # validAmt ensures that invalid order amounts are given 0 value for min_package_req_for_diff_amts
                    validAmt =True
            # updating valid packs used list and minimum number of packages used list
            min_package_req_for_diff_amts[temp_amt]=pcks_used_count if validAmt else 0
            # Adding the last package type to pkgs_used list
            pkgs_used[temp_amt] = new_pkg
        return pkgs_used
                
    

    def printBakeryReceipt(self,orders_dict):
        '''Prints Bakery receipt in provided format
        orders_dict: dict
            contains key = item code (str) value = order amounts(int)
        '''
        order_dict = orders_dict
        # Creating order breakdown dictionary 
        order_breakdown_dict = {k:[] for k,v in self.__items.items()}
        order_total_dict = {k:0 for k,v in self.__items.items()}
        # Determining item breakup for each item ordered
        for k,v in order_dict.items():
            pkgUsedList=self.__determineItemBreakup(self.__valid_package_dict[k],order_dict[k],[0]*(order_dict[k]+1),[0]*(order_dict[k]+1))
            order_breakdown_dict[k],order_total_dict[k] = self.__getPackagesUsedAndTotalForAmt(k,pkgUsedList,order_dict[k])
        print('Your receipt:')    
        # Printing the results
        for k in self.__printOrder:
            if (k in order_dict):
                print ('{} {} ${:g}'.format(order_dict[k],k,order_total_dict[k]))
            
                for i in reversed(sorted(list(set(order_breakdown_dict[k])))):
                    if(order_total_dict[k]==0):
                        print ('{:>17}'.format('Invalid Order'))
                        continue
                    print ('{:>7} x {} ${:g}'.format(order_breakdown_dict[k].count(i),i,self.__items_price_dict[k][i]))
        

        
        
    def __getPackagesUsedAndTotalForAmt(self,k,pkgUsed,amt):
        '''Calculates the package types to be used for order amount as well as the total bill
        Parameters:
        -----------
        k: str
            item code
        pkgUsed: list
            list which now contains the package types for different order amounts
        amt: int
            order amount for k item code
        Returns:
        --------
        pgkUsedList: list
            package types that will be provided to customer 
        
        total: float
            total bill calculated
        '''        
        pgkUsedList = []
        pkg = amt
        total =0
        
        while pkg > 0:
            if(pkgUsed[pkg])==0:
                pgkUsedList.append(0)
                break
            thisPkg = pkgUsed[pkg]
            pgkUsedList.append(thisPkg)
            total = total + self.__items_price_dict[k][thisPkg]
#             print(thisPkg)
            pkg = pkg - thisPkg
        pgkUsedList.sort(reverse=True)
        return pgkUsedList,total
    
    def printMenu(self):
        '''Prints the menu
        '''
        print ("Today's menu: ")
        print ("|{:^20}|{:^20}|{:^50}|".format("Item","Code","Packages"))
        print ("{:-^93}".format(''))
        for k,v in self.__items.items():
            print ("|{:^20}|{:^20}|{:^50}|".format(v,k,str(self.__items_price_dict[k])))
    
    def takeOrder(self, orderList):
        '''Takes order and prints the receipt
        orderList: list
            list of orders of items with elements of list in format 'item_code orderAmount'
        
        '''
        # Check for invalid order - string, negative number, invalid format
        # Creating order dictionary 
        try:
            order_dict = {str.upper(x.split()[1]):int(x.split()[0]) for x in orderList }
           

        except:
            print(' Invalid order format. Must be a list of "UniqueValidItemCode orderAmt(integer)" ')
            return
         # length of your order should be less than or equal to the number of items available in menu
        try:
            assert len(order_dict) <= len(self.__items) ,' Please check your order '
        # check if item code is valid
            for k,v in order_dict.items():
                assert k in self.__items,' Invalid item code '
                #integer check
                assert isinstance(v, int),' Order Amount is invalid. Must be an integer '
                # positive number order check
                assert v>0,' Order amount must be above 0 '
        except AssertionError as e:
            print(e.args[0])
            return
        try:
            self.printBakeryReceipt(order_dict)
        except:
            print(" Bakery malfunction. Please evacuate ")
            raise
            return

        
def main():
    # Initialising bakery with prices and menu
    print("Opening the bakery")
    bakery = Bakery(menu_items = {'VS5':'Vegemite Scroll','MB11':'Blueberry Muffin','CF':'Croissant'},
               menu_price_dict = {'VS5':{3:6.99,5:8.99},'MB11':{2:9.95,5:16.95,8:24.95},'CF':{3:5.95,5:9.95,9:16.99}},
               printOrder = ['VS5','MB11','CF'])
    bakery.printMenu()
    print("\nTest case 1\n")
    print("Order: ")
   
    order_list = ['10 VS5',
             '14 MB11',
             '13 CF']
    for o in order_list:
        print (o)
    print("\nOutput\n")  
    bakery.takeOrder(order_list)
    #-------------------
    print("\nTest case 2 - invalid amounts - some amounts are not available using packages\n")
    print("Order: ")
 
    order_list2 = ['7 VS5',
             '1 MB11',
             '3 CF']
    for o in order_list2:
        print (o)
    print("\nOutput\n")  
    bakery.takeOrder(order_list2)
    #-------------------
    print("\nTest case 3 - negative amounts not accepted\n")
    print("Order: ")
    
    order_list3 = ['-5 VS5',
             '677 MB11',
             '1111 CF']
    for o in order_list3:
        print (o)
    print("\nOutput\n")  
    bakery.takeOrder(order_list3)
    
    print("\nTest case 4 - wrong item code\n")
    print("Order: ")
    
    order_list4 = ['50 Vw4',
             '677 MB11',
             '1111 CF']
    for o in order_list4:
        print (o)
    print("\nOutput\n")  
    bakery.takeOrder(order_list4)
    
    print("\nTest case 5 - invalid format\n")
    print("Order: ")
    
    order_list5 = ['50Vw4',
             '677MB11',
             '1111 CF']
    for o in order_list5:
        print (o)
    print("\nOutput\n")  
    bakery.takeOrder(order_list5)
    s=''
    # Enter your input
    while(str.upper(s)!='X'):
        
        print("\nWould you like to place an order? Type X to exit the bakery or stop order")
        print ("Type the order in the following format and enter an item type only once:")
        print ("orderAmt ItemCode")
        bakery.printMenu()
        order_list0 = []
        for i in range(3):
            s= input()
            if(str.upper(s)=='X'):
                break
            order_list0.append(s)
        if(len(order_list0)==0):
            break
        
        bakery.takeOrder(order_list0)
        s=''
        print("\nWould you like to order more? type X to exit and anything else to continue\n")
        s=input()
    print ("\nThank you for visiting")
    
    
if __name__ == '__main__':
    main()        





