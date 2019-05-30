# Open a file: file
import argparse
import os


#os.getcwd()
parser = argparse.ArgumentParser(description='Do rhetorical analysis')

parser.add_argument('input', help='''Input file name (located in data/) e.g. sonnets.txt''')


args = parser.parse_args()

local_path_input = 'data/' + args.input
abs_path_input = os.path.abspath(local_path_input)
filename, file_extension = os.path.splitext(abs_path_input)
corpus, extension = os.path.splitext(args.input)

local_path_output = filename + '_input' + '.txt'


with open (abs_path_input, 'r', encoding='iso-8859-1') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line

#content = [re.sub(r'[^\x00-\x7f]',r'', x.strip()).strip() + "\n" for x in content] 
#content = [x for x in content if(len(x.split(" ")) > 1)] 
#content = [x.replace(''''d''', '''ed''') for x in content]
#content = [x.replace(''''st''', '''ed''') for x in content]
content = [bytes(x,'utf-8').decode('utf-8','ignore') +"\n" for x in content] 

file = open(local_path_output, 'w+', encoding='utf-8')
one_word = ""
for line in content:
	if(len(line.strip().split()) == 1):
		one_word = one_word + line.strip()
		#print(one_word)
	else:
		file.write("%s\n" % line.strip())
file.close()

abs_path_output = os.path.abspath(local_path_output)

rhet_file = os.path.basename(abs_path_output)

import subprocess
import shlex
import os

def subprocess_cmd(command):
    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    print(proc_stdout)

pwd_dir = os.path.dirname(os.path.realpath(__file__))
output_folder = "%s/data/rhet/repitition" % (pwd_dir)

commands1 = '''
xbuild rhetorica/Rhetorica.sln /p:Configuration=Release /p:Platform=x64;
'''
subprocess_cmd(commands1)



commands2 = '''
mono rhetorica/bin/x64/Release/Rhetorica.exe "%s/" "%s" "{ Anadiplosis: {windowSize: 3} }" "%s/sonnet_anadiplosis";
''' % (pwd_dir, rhet_file, output_folder)


commands3 = '''
xbuild rhetorica/Rhetorica.sln /p:Configuration=Release /p:Platform=x64
mono rhetorica/bin/x64/Release/Rhetorica.exe "" "data/sonnets.txt" "{ Anadiplosis: {windowSize: 3} }" "/usr/project/xtmp/dp195/Poetix18/data/rhet/repitition/sonnet_anadiplosis";
mono rhetorica/bin/x64/Release/Rhetorica.exe "data/sonnets.txt" "{ Anaphora: {windowSize: 3} }" "/usr/project/xtmp/dp195/Poetix18/data/rhet/repitition/sonnet_anaphora";
mono rhetorica/bin/x64/Release/Rhetorica.exe "data/sonnets.txt" "{ Antimetabole: {windowSize: 3} }" "/usr/project/xtmp/dp195/Poetix18/data/rhet/repitition/sonnet_antimetabole";
mono rhetorica/bin/x64/Release/Rhetorica.exe "data/sonnets.txt" "{ Conduplicatio: {windowSize: 3} }" "/usr/project/xtmp/dp195/Poetix18/data/rhet/repitition/sonnet_conduplicatio";
mono rhetorica/bin/x64/Release/Rhetorica.exe "data/sonnets.txt" "{ Epanalepsis: {windowSize: 3} }" "/usr/project/xtmp/dp195/Poetix18/data/rhet/repitition/sonnet_epanalepsis";
mono rhetorica/bin/x64/Release/Rhetorica.exe "data/sonnets.txt" "{ Epistrophe: {windowSize: 3} }" "/usr/project/xtmp/dp195/Poetix18/data/rhet/repitition/sonnet_epistrophe";
mono rhetorica/bin/x64/Release/Rhetorica.exe "data/sonnets.txt" "{ Epizeuxis: {windowSize: 3} }" "/usr/project/xtmp/dp195/Poetix18/data/rhet/repitition/sonnet_epizeuxis";
mono rhetorica/bin/x64/Release/Rhetorica.exe "data/sonnets.txt" "{ Ploce: {windowSize: 3} }" "/usr/project/xtmp/dp195/Poetix18/data/rhet/repitition/sonnet_ploce";
mono rhetorica/bin/x64/Release/Rhetorica.exe "data/sonnets.txt" "{ Polysyndeton: {windowSize: 3} }" "/usr/project/xtmp/dp195/Poetix18/data/rhet/repitition/sonnet_polysyndeton";
mono rhetorica/bin/x64/Release/Rhetorica.exe "data/sonnets.txt" "{ Symploce: {windowSize: 3} }" "/usr/project/xtmp/dp195/Poetix18/data/rhet/repitition/sonnet_symploce";
'''

subprocess_cmd(commands1)
subprocess_cmd(commands2)


