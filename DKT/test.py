from optparse import OptionParser 
parser = OptionParser() 
parser.add_option("--pdcl",'--asd','--qwe','--wer', action="store_true", 
                  dest="pdcl", 
                  default=False, 
                  help="write pdbk data to oracle db") 
parser.add_option("-z", "--zdbk", action="store_true", 
                  dest="zdcl", 
                  default=False, 
                  help="write zdbk data to oracle db") 
parser.add_option("--simple_index",action="store_true", dest="simple_index",\
                 default=True,help="using simple index for training, no onehot,and difference between pages")



(options, args) = parser.parse_args() 
print(args)
print(options)
if options.pdcl==True: 
    print ('pdcl is true') 
if options.zdcl==True: 
    print ('zdcl is true')