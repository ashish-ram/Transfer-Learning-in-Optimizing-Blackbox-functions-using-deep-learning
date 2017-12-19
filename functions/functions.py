import numpy as np
import airfoil.utility as u
import os

class Holder:
    bounds = ((-10, 10), (-10, 10))
    minimas = [(8.05502, 9.6645)]
    fmin = -19.2085
    name = 'holder'
    def __init__(self):
        pass
    def objective(self, xx, *params):
        '''
        https://www.sfu.ca/~ssurjano/holder.html
        INPUT:
        xx = [x1, x2]
    
        Description:
    
        Dimensions: 2 
        The Holder Table function has many local minima, with four global minima. 
    
        Input Domain:
    
        The function is usually evaluated on the square xi ∈ [-10, 10], for all i = 1, 2. 
    
        Global Minimum:
        f(X*) = -19.2085, at X* = (8.05502, 9.6645)
        '''
        x1 = xx[0]
        x2 = xx[1]

        fact1 = np.sin(x1) * np.cos(x2)
        fact2 = np.exp(np.abs(1 - np.sqrt(np.square(x1) + np.square(x2) / np.pi)))

        y = -np.abs(fact1 * fact2)

        return y

class Branin:
    bounds = ((-5,10),(0,15))
    minimas = [(-np.pi, 12.275),(np.pi,2.275),(9.42478,2.475)]
    fmin = 0.397887
    name = 'branin'
    def __init__(self):
        pass

    def objective(self,x, *params):
        '''
        https://www.sfu.ca/~ssurjano/branin.html
        Description:
        f(X) = a(x2-bx1^2+cx1-r)^2 + s(1-t)cos(x1)+s
        Dimensions: 2 
        The Branin, or Branin-Hoo, function has three global minima. 
        The recommended values of a, b, c, r, s and t are: a = 1, b = 5.1 ⁄ (4π2), c = 5 ⁄ π, r = 6, s = 10 and t = 1 ⁄ (8π).
    
        Input Domain:
        This function is usually evaluated on the square x1 ∈ [-5, 10], x2 ∈ [0, 15]. 
    
        Global Minimum:
        f(x*) = 0.397887 at x* =(-π, 12.275),(π,2.275) and (9.42478,2.475)
    
        '''
        x1 = x[0]
        x2 = x[1]
        a = 1.
        b = 5.1 / (4.*np.pi**2)
        c = 5. / np.pi
        r = 6.
        s = 10.
        t = 1. / (8.*np.pi)
        ret  = a*(x2-b*x1**2+c*x1-r)**2+s*(1-t)*np.cos(x1)+s
        #print (ret)
        return ret

class Franke:
    """
    one local min, one global minima, one global maxima
    https://www.sfu.ca/~ssurjano/franke2d.html
    """
    bounds = ((0.0,1.0),(0.0,1.0))
    minimas = []
    name = 'franke'
    def __init__(self):
        pass
    def objective(self, x):
        x1 = x[0]
        x2 = x[1]
        term1 = 0.75 * np.exp(-(9 * x1 - 2) ** 2 / 4 - (9 * x2 - 2) ** 2 / 4)
        term2 = 0.75 * np.exp(-(9 * x1 + 1) ** 2 / 49 - (9 * x2 + 1) / 10)
        term3 = 0.5 * np.exp(-(9 * x1 - 7) ** 2 / 4 - (9 * x2 - 3) ** 2 / 4)
        term4 = -0.2 * np.exp(-(9 * x1 - 4) ** 2 - (9 * x2 - 7) ** 2)

        y = term1 + term2 + term3 + term4
        return -1 * y



class Fun2d:
    bounds = ((-5, 10), (0, 15))
    minimas = [(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)]
    name = 'test 2d'
    def __init__(self):
        pass
    def objective(self,x, *params):
        f = np.cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] + 0.2) * x[0]
        df = np.zeros(2)
        df[0] = -14.5 * np.sin(14.5 * x[0] - 0.3) + 2. * x[0] + 0.2
        df[1] = 2. * x[1] + 0.2
        return f


class Ackley:
    bounds = ((-32.768, 32.768), (-32.768, 32.768))
    minimas = [(0, 0)]
    fmin = 0
    name = 'ackley'

    def __init__(self):
        pass

    def objective(self, x, *params):
        '''
        https://www.sfu.ca/~ssurjano/ackley.html
        '''
        x1 = x[0]
        x2 = x[1]
        a = 20.0
        b = 0.2
        c = 2.0*np.pi

        ret = -a*np.exp(-b*np.sqrt(0.5*(x1**2+x2**2))) - np.exp(0.5*(np.cos(c*x1)+np.cos(c*x2)))+a+np.exp(1)
        # print (ret)
        return ret


class Bukin:
    bounds = ((-15.0, -5.0), (-3.0, 3.0))
    minimas = [(-10, 1)]
    fmin = 0
    name = 'bukin'

    def __init__(self):
        pass

    def objective(self, x, *params):
        '''
        https://www.sfu.ca/~ssurjano/ackley.html
        '''
        x1 = x[0]
        x2 = x[1]
        ret = 100*np.sqrt(np.abs(x2-0.01*x1**2)) + 0.01*np.abs(x1+10)
        # print (ret)
        return ret

class CrossIT:
    bounds = ((-10, 10), (-10, 10))
    minimas = [(1.3491,1.3491),(1.3491,-1.3491),(-1.3491,-1.3491),(-1.3491,1.3491)]
    fmin = -2.06261
    name = 'crossIT'

    def __init__(self):
        pass

    def objective(self, x, *params):
        '''
        https://www.sfu.ca/~ssurjano/ackley.html
        '''
        x1 = x[0]
        x2 = x[1]

        fact1 = np.sin(x1) * np.sin(x2)
        fact2 = np.exp(np.abs(100 - np.sqrt(x1 ** 2 + x2 ** 2) / np.pi))

        y = -0.0001 * np.power(np.abs(fact1 * fact2) + 1 , 0.1)

        # print (ret)
        return y

class Shubert:
    bounds = ((-4, 2), (-4, 2))
    minimas = None
    fmin = -186.7309088
    name = 'shubert'

    def __init__(self):
        pass

    def objective(self, x, *params):
        '''
        https://www.sfu.ca/~ssurjano/Code/shubertm.html
        http://profesores.elo.utfsm.cl/~tarredondo/info/soft-comp/functions/node28.html
        '''
        x1 = x[0]
        x2 = x[1]
        sum1 = 0
        sum2 = 0

        for i in range(6):
            new1 = i*np.cos((i+1)*x1+i)
            new2 = i*np.cos((i+1)*x2+i)
            sum1 = sum1+new1
            sum2 = sum2+new2
        return sum1*sum2


class Airfoil:
    #bounds=((0.2,-1.0),(0.2,-1.0)) => used for sampling
    #bounds = ((-1.0,0.2), (-1.0,0.2))
    bounds = ((-1.0, 1.0), (-1.0, 1.0))
    minima = None
    fmin = None
    name = 'airfoil'
    foil_num = 1

    def __init__(self, *params):
        if params is not None:
            self.alfa, self.Ma, self.Re, self.foil_path, self.graphics = params
            self.params = self.alfa, self.Ma, self.Re, self.foil_path

    def expand_ys_2d_6d(self,ys):
        tmp = [0.05, 0.1, 0.2]
        tmp.append(ys[0])
        tmp.append(ys[1])
        tmp.append(-0.05)
        ys = np.array(tmp)
        # print('expanded',ys)
        return ys

    def objective(self, x, *params):
        '''
        '''
        print("Point Evaluated: ",x)
        x1 = x[0]
        x2 = x[1]
        control_points_list = u.prepare_controls(self.expand_ys_2d_6d(x))
        ld = u.xfoil(control_points_list, 'temp_foil', self.params, method='bezier', timeout=5, graphics=self.graphics)
        os.remove(self.foil_path+'temp_foil.log')
        os.remove(self.foil_path+'temp_foil.dat')
        return -ld

    def objective_opt(self, x, *params):
        '''
        use this objective for any optimization task. It returns LD = 20 incase of an invalid instance of airfoil
        '''
        print("Point Evaluated: ",x)
        x1 = x[0]
        x2 = x[1]
        control_points_list = u.prepare_controls(self.expand_ys_2d_6d(x))
        ld = u.xfoil(control_points_list, 'temp_foil', self.params, method='bezier', timeout=5, graphics=self.graphics)
        os.remove(self.foil_path+'temp_foil.log')
        os.remove(self.foil_path+'temp_foil.dat')
        if ld==-float('inf'):
            return 20   ## just a random number close to function maxima(ceiling)
        return -ld

    def objective_6D(self, x, *params):
        '''
        objective function for 6D dimensional model
        '''
        print("Point Evaluated#####: ", x)
        print(self.foil_num)
        control_points_list = u.prepare_controls(x)
        ld = u.xfoil(control_points_list, str(self.foil_num), self.params, method='bezier', timeout=15, graphics=self.graphics)
        #os.remove(self.foil_path + 'temp_foil.log')
        #os.remove(self.foil_path + 'temp_foil.dat')
        if ld == -float('inf'):
            self.foil_num = self.foil_num + 1
            return 20  ## just a random number close to function maxima(ceiling)
        self.foil_num = self.foil_num +1
        return -ld

if __name__ =="__main__":
    import airfoil.utility as u
    test_dir = r'./test_DIR_/'

    u.make_sure_path_exists(test_dir)

    #params = 5, 0.2, 5e005, test_dir, True
    funObj = Airfoil(5, 0.77971043, 5e005, test_dir, True)
    #ld = funObj.objective_opt(np.array([-0.616162 , 0.656566	]))  # use for max LD test
    #ld = funObj.objective(np.array([-1, 1]))   # test for bad shape foil
    for i in range(1):
        ld = funObj.objective_6D(np.array([  7.13340921e-02,   2.26597498e-02,   2.19496322e-02,
        -5.76227027e-02,   4.45963135e-02,  -3.60694583e-02,
         3.00000000e+00,   7.79710428e-01,   5.00000000e+05])
)  # use for 6D LD test
        print("neg. LD = ", ld)



