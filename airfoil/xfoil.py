"""
http://stackoverflow.com/questions/1191374/using-module-subprocess-with-timeout
"""
import subprocess, threading
#import airfoil.utility
import os

class Error(Exception):
    pass
class TerminationException(Error):
    def __init__(self, path, airfoil_file_name):
        self.path = path
        self.airfoil_file_name = airfoil_file_name
    def handle(self):
        error_log_file = open(self.path + "failed.log", 'a')
        error_log_file.write(self.airfoil_file_name + " : Convergence Failed"'\n')
        error_log_file.close()


class Command(object):
    def __init__(self,airfoil_file_name, params):
        alfa,Ma, Re, path = params
        self.cmd = r'/usr/bin/xfoil'
        self.process = None
        self.airfoil_file_name = airfoil_file_name
        self.path = path
        self.alfa = alfa
        self.Ma = Ma
        self.Re = Re
        self.xfoilpath = r'/usr/bin/xfoil'
    def __del__(self):
        print('Threading objected deleted')

    def run(self, timeout, graphics):
        def target():
            #print ('Thread started for foil : {}'.format(self.airfoil_file_name))
            try:
             f =open(self.path+self.airfoil_file_name+'.dat', 'r')
             f.close()
            except FileNotFoundError as e:
                error_log_file = open(self.path+"failed.log",'a')
                error_log_file.write(self.airfoil_file_name+" : file not found"'\n')
                error_log_file.close()
            else:
                self.process = subprocess.Popen(self.cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True)

                out, err = self.process.communicate(
                                        #"plop\n"
                                        #"g\n"
                                        #" \n"
                                        "load {}\n"
                                        "foil{}\n"
                                        "oper\n"
                                        "visc {}\n"
                                        "M {}\n"
                                        "ITER\n"
                                        "500\n"
                                        "pacc\n"
                                        "{}\n"
                                        " \n"
                                        #"alfa {}"
                                        # "aseq {} {} {}\n"
                                        "aseq 0 {} 1\n"
                                        # "hard"
                                        " \n"
                                        "quit\n".format(
                                                        self.path + self.airfoil_file_name + '.dat',
                                                        self.airfoil_file_name,
                                                        self.Re,
                                                        self.Ma,
                                                        self.path + self.airfoil_file_name + '.log',
                                                        self.alfa)
                                                )                 
        def target_no_graphics():
            #print ('Thread started for foil : {}'.format(self.airfoil_file_name))
            try:
             f =open(self.path+self.airfoil_file_name+'.dat', 'r')
             f.close()
            except FileNotFoundError as e:
                error_log_file = open(self.path+"failed.log",'a')
                error_log_file.write(self.airfoil_file_name+" : file not found"'\n')
                error_log_file.close()
            else:
                self.process = subprocess.Popen(self.cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True)

                out, err = self.process.communicate(
                                        "plop\n"
                                        "g\n"
                                        " \n"
                                        "load {}\n"
                                        "foil{}\n"
                                        "oper\n"
                                        "visc {}\n"
                                        "M {}\n"
                                        "ITER\n"
                                        "300\n"
                                        "pacc\n"
                                        "{}\n"
                                        " \n"
                                        #"alfa {}"
                                        # "aseq {} {} {}\n"
                                        "aseq 0 {} 1\n"
                                        # "hard"
                                        " \n"
                                        "quit\n".format(
                                                        self.path + self.airfoil_file_name + '.dat',
                                                        self.airfoil_file_name,
                                                        self.Re,
                                                        self.Ma,
                                                        self.path + self.airfoil_file_name + '.log',
                                                        self.alfa)
                                                )                 
                #print(out)
                #print ('Thread finished for foil: {}'.format(self.airfoil_file_name))
        if graphics==True:
            thread = threading.Thread(target=target)
        else:
            thread = threading.Thread(target=target_no_graphics)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            #print ('Terminating process')
            self.process.terminate()
            thread.join()
        try:
            retutncode = self.process.returncode
            if retutncode==-15:
                raise TerminationException(self.path, self.airfoil_file_name)
        except TerminationException as te:
            te.handle()
        except AttributeError as e:
            error_log_file = open(self.path + "failed.log", 'a')
            error_log_file.write(self.airfoil_file_name +": xfoil not executed" '\n')
            error_log_file.close()
        #return self.getLDfromLog()

    def getLDfromLog(self):
        alfa = self.alfa
        filename = self.path + self.airfoil_file_name + ".log"

        f = open(filename, 'r')
        flines = f.readlines()
        LD = dict()
        for i in range(12, len(flines)):
            #print flines[i]
            words = str.split(flines[i])
            alfa = float(words[0])
            LD[alfa] = float(words[1]) / float(words[2])
        print("LD at different Alfa for current Airfoil: ",LD)
        if self.alfa in LD.keys():
            print("returning LD at alfa asked: ", LD)
            return LD[self.alfa]
        else:
            print("current foil at this alfa doesn't converge => penalty LD = -INF")
            return -float('inf')









if __name__ == "__main__":
    pass

