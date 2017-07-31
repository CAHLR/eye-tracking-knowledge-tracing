import argparse
import sys

parser = argparse.ArgumentParser(description='test parsing arguments')

parser.add_argument('--batch_size',type = int)


print (sys.argv)
# arg = parser.parse_args(sys.argv[1:])
arg = parser.parse_args(sys.argv[1:])
print (arg)

# print parser.print_help()
