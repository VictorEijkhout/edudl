#!/usr/bin/env python3

import re
import sys

def usage():
    print("Usage: configure.py [ --blis BLIS_DIR ]")
    sys.exit(0)

args = sys.argv[1:]
if len(args)==0:
    print("No Make.inc generated")
    usage()

##
## parse commandline arguments for
## --blis : blis include directory
##

makeinc = open("Make.inc","w")
while len(args)>0:
    a,args = args[0],args[1:]
    if not re.match(r'--',a):
        usage()
    elif re.match(r'--blis',a):
        if len(args)==0:
            print("Missing blis argument")
            sys.exit(1)
        else:
            b,args = args[0],args[1:]
            if re.match(r'--',b):
                print(f"Probably missing blis argument: <<{b}>>")
                sys.exit(1)
            makeinc.write(f"""CXX=clang++ -std=c++17 -fopenmp
			USE_BLIS=1
BLIS_INC_DIR={b}/include
BLIS_LIB_DIR={b}/lib
""")
    else:
        print(f"Unknown option {a}")
        sys.exit(1)
        
