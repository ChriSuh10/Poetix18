# Open a file: file
import argparse
import os


#os.getcwd()
parser = argparse.ArgumentParser(description='Do rhetorical analysis')

parser.add_argument('-i', '--input', help='Input file name located in data/', required=True)

args = parser.parse_args()

local_path_input = 'data/' + args.i
abs_path_input = os.path.abspath(local_path_input)
filename, file_extension = os.path.splitext(abs_path_input)

local_path_output = 'data/' + filename + '_input' + '.txt'


with open (abs_path_input, 'r', encoding='iso-8859-1') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line

#content = [re.sub(r'[^\x00-\x7f]',r'', x.strip()).strip() + "\n" for x in content] 
#content = [x for x in content if(len(x.split(" ")) > 1)] 
#content = [x.replace(''''d''', '''ed''') for x in content]
#content = [x.replace(''''st''', '''ed''') for x in content]
content = [bytes(x,'utf-8').decode('utf-8','ignore') +"\n" for x in content] 

file = open(local_path_output, 'w', encoding='utf-8')
one_word = ""
for line in content:
	if(len(line.strip().split()) == 1):
		one_word = one_word + line.strip()
		print(one_word)
	else:
		file.write("%s\n" % line.strip())
file.close()

abs_path_output = os.path.abspath(local_path_output)

